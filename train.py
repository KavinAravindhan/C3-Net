import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import json
from datetime import datetime
import wandb
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from data.dataset import MIMICEyeDataset, collate_fn
from models.teacher import MultimodalTeacher

load_dotenv()


class C3NetTrainer:
    """
    Training system for C3-Net - Teacher Model Only.

    Modality-aware: controlled by config['model']['modality']
        image_only      — image encoder only
        image_gaze      — image + gaze encoders
        image_text      — image + text encoders
        image_gaze_text — all three encoders

    Staged BERT training (for text-containing modalities):
        Stage 1 (epochs 1 to bert_freeze_epochs):  BERT frozen
        Stage 2 (bert_freeze_epochs+1 onwards):    BERT unfrozen with 10x lower LR
        bert_freeze_epochs=0 means BERT unfrozen from epoch 1

    Loss: Weighted cross-entropy (handles class imbalance)
    Metrics: Accuracy, Precision, Recall, F1, AUC
    Best model saved by: val AUC
    """

    def __init__(self, config, run_name=None):
        self.config = config

        self.modality           = config['model'].get('modality', 'image_gaze_text')
        self.bert_freeze_epochs = config['model'].get('bert_freeze_epochs', 5)
        self.uses_text          = self.modality in ('image_text', 'image_gaze_text')
        self.uses_gaze          = self.modality in ('image_gaze', 'image_gaze_text')
        self.bert_unfrozen      = False

        if config['training']['device'] == 'cuda' and torch.cuda.is_available():
            gpu_id = config['training']['gpu_id']
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        print("="*80)
        print("Initializing C3-Net Teacher Training")
        print("="*80)
        print(f"Modality:           {self.modality}")
        if self.uses_text:
            if self.bert_freeze_epochs == 0:
                print(f"BERT training:      Unfrozen from epoch 1")
            else:
                print(f"BERT freeze epochs: {self.bert_freeze_epochs}")

        print("\n1. Loading MultimodalTeacher...")
        self.teacher = MultimodalTeacher(config=config).to(self.device)

        class_weights = torch.tensor([1.0/0.33, 1.0/0.67]).to(self.device)
        class_weights = class_weights / class_weights.sum() * 2.0
        print(f"\n2. Class weights: Normal={class_weights[0]:.3f}, Abnormal={class_weights[1]:.3f}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.lr      = config['training']['learning_rate']
        self.bert_lr = self.lr * 0.1

        self.non_bert_params = [
            p for name, p in self.teacher.named_parameters()
            if 'text_encoder' not in name and p.requires_grad
        ]
        self.optimizer = optim.AdamW(
            self.non_bert_params,
            lr=self.lr,
            weight_decay=config['training']['weight_decay']
        )

        if self.uses_text and self.bert_freeze_epochs == 0:
            self.teacher.unfreeze_bert()
            self.optimizer.add_param_group({
                'params': [p for p in self.teacher.text_encoder.parameters()],
                'lr':     self.bert_lr
            })
            self.bert_unfrozen = True

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=1e-6
        )

        self.best_val_acc  = 0.0
        self.best_val_f1   = 0.0
        self.best_val_auc  = 0.0
        self.best_epoch    = 0
        self.best_metrics  = {}
        self.train_history = []
        self.val_history   = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_name is None:
            run_name = f"teacher_{self.modality}_{timestamp}"

        wandb.init(
            entity="ai4vs",
            project="c3_net",
            name=run_name,
            config={
                "model":              "MultimodalTeacher",
                "modality":           self.modality,
                "learning_rate":      self.lr,
                "bert_lr":            self.bert_lr if self.uses_text else "N/A",
                "batch_size":         config['training']['batch_size'],
                "num_epochs":         config['training']['num_epochs'],
                "weight_decay":       config['training']['weight_decay'],
                "dropout":            config['model']['dropout'],
                "bert_freeze_epochs": self.bert_freeze_epochs if self.uses_text else "N/A",
                "lambda_gaze_pred":   config['training'].get('lambda_gaze_pred', 0.5) if self.uses_gaze else "N/A",
                "image_encoder":      config['model']['image_encoder']['type'],
                "class_weights":      f"Normal={class_weights[0]:.3f}, Abnormal={class_weights[1]:.3f}",
            }
        )

        total_params     = sum(p.numel() for p in self.teacher.parameters())
        trainable_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        bert_status      = "unfrozen from epoch 1" if (self.uses_text and self.bert_freeze_epochs == 0) \
                           else f"frozen for {self.bert_freeze_epochs} epochs" if self.uses_text \
                           else "N/A (no text encoder)"
        print(f"\n✓ Teacher initialized")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  BERT status:      {bert_status}")

    # ------------------------------------------------------------------
    def _get_batch_inputs(self, batch):
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)

        if self.uses_gaze:
            gaze_heatmaps    = batch['gaze_heatmaps'].to(self.device)
            gaze_sequences   = batch['gaze_sequences'].to(self.device)
            gaze_seq_lengths = batch['gaze_seq_lengths']
        else:
            gaze_heatmaps    = None
            gaze_sequences   = None
            gaze_seq_lengths = None

        if self.uses_text:
            text_token_ids       = batch['text_token_ids'].to(self.device)
            text_attention_masks = batch['text_attention_masks'].to(self.device)
        else:
            text_token_ids       = None
            text_attention_masks = None

        return images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
               text_token_ids, text_attention_masks, labels

    # ------------------------------------------------------------------
    def _compute_metrics(self, all_labels, all_preds, all_probs):
        acc  = np.mean(np.array(all_preds) == np.array(all_labels))
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

    # ------------------------------------------------------------------
    def train_epoch(self, train_loader, epoch):
        self.teacher.train()

        total_loss = 0.0
        all_labels = []
        all_preds  = []
        all_probs  = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch in pbar:
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
            text_token_ids, text_attention_masks, labels = self._get_batch_inputs(batch)

            self.optimizer.zero_grad()

            logits, _ = self.teacher(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks
            )

            loss = self.criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
            self.optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_labels += labels.cpu().tolist()
            all_preds  += preds.cpu().tolist()
            all_probs  += probs.detach().cpu().tolist()

            running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{running_acc*100:.2f}%'})

            wandb.log({"batch/train_loss": loss.item()})

        epoch_loss    = total_loss / len(train_loader)
        epoch_metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        return epoch_loss, epoch_metrics

    # ------------------------------------------------------------------
    def validate(self, val_loader, epoch):
        self.teacher.eval()

        total_loss = 0.0
        all_labels = []
        all_preds  = []
        all_probs  = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for batch in pbar:
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
                text_token_ids, text_attention_masks, labels = self._get_batch_inputs(batch)

                logits, _ = self.teacher(
                    images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                    text_token_ids, text_attention_masks
                )

                loss  = self.criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                all_labels += labels.cpu().tolist()
                all_preds  += preds.cpu().tolist()
                all_probs  += probs.cpu().tolist()

                running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{running_acc*100:.2f}%'})

        epoch_loss    = total_loss / len(val_loader)
        epoch_metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        return epoch_loss, epoch_metrics

    # ------------------------------------------------------------------
    def evaluate_test(self, test_loader):
        """
        Run evaluation on the test set using the best saved model.

        Loads best_model.pth from save_dir (set during training), runs inference,
        and returns metrics + raw predicted probabilities for DeLong's test.

        Args:
            test_loader: DataLoader for test split

        Returns:
            metrics: dict with accuracy, precision, recall, f1, auc
            all_probs: list of predicted probabilities (class=1) — needed for DeLong's test
            all_labels: list of ground truth labels
        """
        self.teacher.eval()

        all_labels = []
        all_preds  = []
        all_probs  = []

        pbar = tqdm(test_loader, desc="Test Evaluation")

        with torch.no_grad():
            for batch in pbar:
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
                text_token_ids, text_attention_masks, labels = self._get_batch_inputs(batch)

                logits, _ = self.teacher(
                    images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                    text_token_ids, text_attention_masks
                )

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_labels += labels.cpu().tolist()
                all_preds  += preds.cpu().tolist()
                all_probs  += probs.cpu().tolist()

        metrics = self._compute_metrics(all_labels, all_preds, all_probs)

        print(f"\nTest Set Results:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1:        {metrics['f1']*100:.2f}%")
        print(f"  AUC:       {metrics['auc']:.4f}")

        return metrics, all_probs, all_labels

    # ------------------------------------------------------------------
    def _unfreeze_bert(self):
        self.teacher.unfreeze_bert()
        self.optimizer.add_param_group({
            'params': [p for p in self.teacher.text_encoder.parameters()],
            'lr':     self.bert_lr
        })
        self.bert_unfrozen = True
        trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print(f"\n  ✓ BERT unfrozen. Total trainable params: {trainable:,}")

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints',
              save_periodic_checkpoints=True, finish_wandb=True):
        """
        Main training loop.

        Returns:
            best_metrics: dict with best val metrics (selected by AUC)
        """
        os.makedirs(save_dir, exist_ok=True)

        has_stages = self.uses_text and self.bert_freeze_epochs > 0

        print("\n" + "="*80)
        print("Starting Teacher Training")
        print("="*80)
        print(f"Total epochs:   {num_epochs}")
        print(f"Train samples:  {len(train_loader.dataset)}")
        print(f"Val samples:    {len(val_loader.dataset)}")
        print(f"Batch size:     {train_loader.batch_size}")
        print(f"Modality:       {self.modality}")
        if has_stages:
            print(f"Stage 1:        Epochs 1-{self.bert_freeze_epochs}  (BERT frozen)")
            print(f"Stage 2:        Epochs {self.bert_freeze_epochs+1}+  (BERT unfrozen, LR={self.bert_lr:.2e})")
        elif self.uses_text and self.bert_freeze_epochs == 0:
            print(f"BERT training:  Unfrozen from epoch 1 (LR={self.bert_lr:.2e})")
        else:
            print(f"BERT:           Not used for modality '{self.modality}'")

        for epoch in range(num_epochs):

            if self.uses_text and not self.bert_unfrozen and epoch == self.bert_freeze_epochs:
                print(f"\n{'='*80}")
                print(f"STAGE 2: Unfreezing BERT")
                print(f"{'='*80}")
                self._unfreeze_bert()

            if has_stages:
                stage = '1' if epoch < self.bert_freeze_epochs else '2'

            print(f"\n{'='*80}")
            if has_stages:
                print(f"Epoch {epoch+1}/{num_epochs}  |  Stage {stage}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*80)

            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            val_loss,   val_metrics   = self.validate(val_loader, epoch)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  {'Metric':<12} {'Train':>10} {'Val':>10}")
            print(f"  {'-'*34}")
            print(f"  {'Loss':<12} {train_loss:>10.4f} {val_loss:>10.4f}")
            print(f"  {'Accuracy':<12} {train_metrics['accuracy']*100:>9.2f}% {val_metrics['accuracy']*100:>9.2f}%")
            print(f"  {'Precision':<12} {train_metrics['precision']*100:>9.2f}% {val_metrics['precision']*100:>9.2f}%")
            print(f"  {'Recall':<12} {train_metrics['recall']*100:>9.2f}% {val_metrics['recall']*100:>9.2f}%")
            print(f"  {'F1':<12} {train_metrics['f1']*100:>9.2f}% {val_metrics['f1']*100:>9.2f}%")
            print(f"  {'AUC':<12} {train_metrics['auc']:>10.4f} {val_metrics['auc']:>10.4f}")
            print(f"  LR: {current_lr:.2e}")

            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1  = val_metrics['f1']
                self.best_val_auc = val_metrics['auc']
                self.best_epoch   = epoch + 1
                self.best_metrics = val_metrics.copy()
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'), epoch, val_metrics
                )
                print(f"\n  ✓ New best model saved! (Val AUC: {val_metrics['auc']:.4f})")

            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,

                "train/loss":      train_loss,
                "train/accuracy":  train_metrics['accuracy'] * 100,
                "train/precision": train_metrics['precision'] * 100,
                "train/recall":    train_metrics['recall'] * 100,
                "train/f1":        train_metrics['f1'] * 100,
                "train/auc":       train_metrics['auc'],

                "val/loss":        val_loss,
                "val/accuracy":    val_metrics['accuracy'] * 100,
                "val/precision":   val_metrics['precision'] * 100,
                "val/recall":      val_metrics['recall'] * 100,
                "val/f1":          val_metrics['f1'] * 100,
                "val/auc":         val_metrics['auc'],

                "val/best_auc":      self.best_val_auc,
                "val/best_accuracy": self.best_val_acc * 100,
                "val/best_f1":       self.best_val_f1 * 100,

                "overfitting_gap": (train_metrics['accuracy'] - val_metrics['accuracy']) * 100,
            })

            self.train_history.append({'epoch': epoch+1, 'loss': train_loss, **train_metrics})
            self.val_history.append({'epoch': epoch+1, 'loss': val_loss, **val_metrics})

            if save_periodic_checkpoints and (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch, val_metrics
                )

        self.save_training_history(save_dir)
        # wandb.finish()
        if finish_wandb:
            wandb.finish()

        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"\nBest Model — Epoch {self.best_epoch}:")
        print(f"  Accuracy:  {self.best_metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {self.best_metrics['precision']*100:.2f}%")
        print(f"  Recall:    {self.best_metrics['recall']*100:.2f}%")
        print(f"  F1:        {self.best_metrics['f1']*100:.2f}%")
        print(f"  AUC:       {self.best_metrics['auc']:.4f}")
        print(f"\nCheckpoints saved to: {save_dir}")
        print("="*80)

        return self.best_metrics

    # ------------------------------------------------------------------
    def save_checkpoint(self, path, epoch, val_metrics):
        torch.save({
            'epoch':                epoch,
            'modality':             self.modality,
            'teacher_state_dict':   self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics':          val_metrics,
        }, path)

    # ------------------------------------------------------------------
    def save_training_history(self, save_dir):
        history = {
            'modality':     self.modality,
            'train':        self.train_history,
            'val':          self.val_history,
            'best_epoch':   self.best_epoch,
            'best_metrics': self.best_metrics,
        }
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


# ==============================================================================
def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Loading datasets...")
    train_dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='train',
        config=config
    )
    val_dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='val',
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    trainer = C3NetTrainer(config)

    modality       = config['model'].get('modality', 'image_gaze_text')
    timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(
        config['training']['checkpoint_dir'], f'teacher_{modality}_{timestamp}'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=checkpoint_dir,
        save_periodic_checkpoints=True
    )


if __name__ == '__main__':
    main()