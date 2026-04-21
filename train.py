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
from models.medgemma_model import MedGemmaModel

load_dotenv()


# ==============================================================================
# Utility
# ==============================================================================

def compute_class_weights(train_dataset, device):
    """
    Compute class weights using He & Garcia (2009) formula: w_c = 1 / N_c.

    Args:
        train_dataset: MIMICEyeDataset with .samples list
        device:        torch.device

    Returns:
        class_weights: Tensor [2] — [w_normal, w_abnormal]
    """
    labels     = [s['label'] for s in train_dataset.samples]
    n_normal   = labels.count(0)
    n_abnormal = labels.count(1)

    w_normal   = 1.0 / n_normal
    w_abnormal = 1.0 / n_abnormal

    class_weights = torch.tensor([w_normal, w_abnormal], dtype=torch.float32).to(device)

    print(f"  Class counts  — Normal: {n_normal}, Abnormal: {n_abnormal}")
    print(f"  Class weights — Normal: {w_normal:.6f}, Abnormal: {w_abnormal:.6f}")

    return class_weights


# ==============================================================================
# C3NetTrainer — BioClinicalBERT + BioGPT pipeline (use_medgemma: false)
# ==============================================================================

class C3NetTrainer:
    """
    Training system for C3-Net Teacher Model (BioClinicalBERT + BioGPT pipeline).
    Only used when config['model']['use_medgemma'] is False.

    Modality-aware: controlled by config['model']['modality']
        image_only      — image encoder only
        image_gaze      — image + gaze encoders
        image_text      — image + text encoders
        image_gaze_text — all three encoders

    Staged BERT training (for text-containing modalities):
        Stage 1 (epochs 1 to bert_freeze_epochs):  BERT frozen
        Stage 2 (bert_freeze_epochs+1 onwards):    BERT unfrozen with 10x lower LR
        bert_freeze_epochs=0 means BERT unfrozen from epoch 1

    Loss: Weighted cross-entropy (He & Garcia 2009 — w_c = 1/N_c)
    Metrics: Accuracy, Precision, Recall, F1, AUC
    Best model saved by: val AUC
    """

    def __init__(self, config, train_dataset, run_name=None):
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
        print("Initializing C3-Net Teacher Training  [BioClinicalBERT + BioGPT]")
        print("="*80)
        print(f"Modality: {self.modality}")
        if self.uses_text:
            if self.bert_freeze_epochs == 0:
                print(f"BERT training: Unfrozen from epoch 1")
            else:
                print(f"BERT freeze epochs: {self.bert_freeze_epochs}")

        print("\n1. Loading MultimodalTeacher...")
        self.teacher = MultimodalTeacher(config=config).to(self.device)

        print("\n2. Computing class weights (He & Garcia 2009)...")
        class_weights  = compute_class_weights(train_dataset, self.device)
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

        self.best_val_auc  = 0.0
        self.best_val_acc  = 0.0
        self.best_val_f1   = 0.0
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
                "pipeline":           "BioClinicalBERT+BioGPT",
                "modality":           self.modality,
                "learning_rate":      self.lr,
                "bert_lr":            self.bert_lr if self.uses_text else "N/A",
                "batch_size":         config['training']['batch_size'],
                "num_epochs":         config['training']['num_epochs'],
                "weight_decay":       config['training']['weight_decay'],
                "dropout":            config['model']['dropout'],
                "bert_freeze_epochs": self.bert_freeze_epochs if self.uses_text else "N/A",
                "image_encoder":      config['model']['image_encoder']['type'],
            }
        )

        total_params     = sum(p.numel() for p in self.teacher.parameters())
        trainable_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print(f"\n✓ Teacher initialized")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    # ------------------------------------------------------------------
    def _get_batch_inputs(self, batch):
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)

        gaze_heatmaps    = batch['gaze_heatmaps'].to(self.device)    if self.uses_gaze else None
        gaze_sequences   = batch['gaze_sequences'].to(self.device)   if self.uses_gaze else None
        gaze_seq_lengths = batch['gaze_seq_lengths']                 if self.uses_gaze else None
        text_token_ids       = batch['text_token_ids'].to(self.device)      if self.uses_text else None
        text_attention_masks = batch['text_attention_masks'].to(self.device) if self.uses_text else None

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
        Returns metrics + raw probabilities for DeLong's test.
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
        Main training loop. Best model saved by val AUC.
        """
        os.makedirs(save_dir, exist_ok=True)

        has_stages = self.uses_text and self.bert_freeze_epochs > 0

        print("\n" + "="*80)
        print("Starting Teacher Training  [BioClinicalBERT + BioGPT]")
        print("="*80)
        print(f"Total epochs:  {num_epochs}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples:   {len(val_loader.dataset)}")
        print(f"Batch size:    {train_loader.batch_size}")
        print(f"Modality:      {self.modality}")
        if has_stages:
            print(f"Stage 1:       Epochs 1-{self.bert_freeze_epochs}  (BERT frozen)")
            print(f"Stage 2:       Epochs {self.bert_freeze_epochs+1}+  (BERT unfrozen, LR={self.bert_lr:.2e})")

        for epoch in range(num_epochs):

            if self.uses_text and not self.bert_unfrozen and epoch == self.bert_freeze_epochs:
                print(f"\n{'='*80}")
                print(f"STAGE 2: Unfreezing BERT")
                print(f"{'='*80}")
                self._unfreeze_bert()

            stage = '1' if (has_stages and epoch < self.bert_freeze_epochs) else '2' if has_stages else None

            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}" + (f"  |  Stage {stage}" if stage else ""))
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
                "epoch":           epoch + 1,
                "learning_rate":   current_lr,
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
# MedGemmaTrainer — MedGemma pipeline (use_medgemma: true)
# ==============================================================================

class MedGemmaTrainer:
    """
    Two-stage training system for MedGemma pipeline.
    Only used when config['model']['use_medgemma'] is True.

    Stage 1 — GazePredictor:
        Trains GazePredictor supervised on real gaze heatmaps (MSE loss).
        All other components frozen. Fast convergence (~5 epochs).

    Stage 2 — Full model:
        Trains MedGemma decoder + classification head jointly using real gaze.
        GazePredictor frozen in this stage.
        Loss: lambda_gen * generation_loss + lambda_cls * classification_loss
        Best model saved by val AUC (classification) since generation metrics
        require full decoding which is expensive per epoch.

    Inference:
        GazePredictor generates synthetic heatmap from image — no real gaze needed.
        This is the deployment scenario and primary ablation comparison point.
    """

    def __init__(self, config, train_dataset, run_name=None):
        self.config   = config
        self.modality = config['model'].get('modality', 'image_gaze_text')

        if config['training']['device'] == 'cuda' and torch.cuda.is_available():
            gpu_id = config['training']['gpu_id']
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        print("="*80)
        print("Initializing C3-Net MedGemma Training")
        print("="*80)

        print("\n1. Loading MedGemmaModel...")
        self.model = MedGemmaModel(config=config).to(self.device)

        print("\n2. Computing class weights (He & Garcia 2009)...")
        class_weights      = compute_class_weights(train_dataset, self.device)
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Loss weights from config
        self.lambda_gen = config['training'].get('lambda_gen', 0.7)
        self.lambda_cls = config['training'].get('lambda_cls', 0.3)

        # Learning rates
        self.base_lr        = config['training']['learning_rate']
        self.medgemma_lr    = config['training'].get('medgemma_lr', 0.00002)
        self.gaze_pred_lr   = config['training'].get('gaze_predictor_lr', 0.0001)

        # Stage epoch counts
        self.gaze_pred_epochs = config['training'].get('gaze_predictor_epochs', 5)
        self.medgemma_epochs  = config['training'].get('medgemma_epochs', 20)

        # Tracking
        self.best_val_auc  = 0.0
        self.best_val_acc  = 0.0
        self.best_val_f1   = 0.0
        self.best_epoch    = 0
        self.best_metrics  = {}
        self.train_history = []
        self.val_history   = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_name is None:
            run_name = f"medgemma_{self.modality}_{timestamp}"

        wandb.init(
            entity="ai4vs",
            project="c3_net",
            name=run_name,
            config={
                "model":               "MedGemmaModel",
                "pipeline":            "MedGemma",
                "modality":            self.modality,
                "base_lr":             self.base_lr,
                "medgemma_lr":         self.medgemma_lr,
                "gaze_predictor_lr":   self.gaze_pred_lr,
                "gaze_pred_epochs":    self.gaze_pred_epochs,
                "medgemma_epochs":     self.medgemma_epochs,
                "lambda_gen":          self.lambda_gen,
                "lambda_cls":          self.lambda_cls,
                "batch_size":          config['training']['batch_size'],
                "weight_decay":        config['training']['weight_decay'],
                "use_gaze_conditioning": config['decoder'].get('use_gaze_conditioning', True),
            }
        )

        trainable, total = self.model.get_trainable_params()
        print(f"\n✓ MedGemmaModel initialized")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  lambda_gen: {self.lambda_gen}  |  lambda_cls: {self.lambda_cls}")

    # ------------------------------------------------------------------
    def _get_batch_inputs(self, batch):
        images           = batch['images'].to(self.device)
        labels           = batch['labels'].to(self.device)
        gaze_heatmaps    = batch['gaze_heatmaps'].to(self.device)
        gaze_sequences   = batch['gaze_sequences'].to(self.device)
        gaze_seq_lengths = batch['gaze_seq_lengths']
        report_texts     = batch['report_texts']
        return images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, labels, report_texts

    # ------------------------------------------------------------------
    def _tokenize_reports(self, report_texts):
        """
        Tokenize raw transcription strings for MedGemma decoder input.
        Returns input_ids, attention_mask, and labels (input_ids shifted for CE loss).
        """
        max_length = self.config['model']['medgemma']['max_length']
        encoding   = self.model.tokenizer(
            report_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids      = encoding['input_ids'].to(self.device)       # [B, max_length]
        attention_mask = encoding['attention_mask'].to(self.device)  # [B, max_length]

        # Labels: same as input_ids but padding positions set to -100
        labels = input_ids.clone()
        labels[labels == self.model.tokenizer.pad_token_id] = -100   # [B, max_length]

        return input_ids, attention_mask, labels

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
    # Stage 1 — GazePredictor training
    # ------------------------------------------------------------------

    def train_gaze_predictor(self, train_loader, val_loader, save_dir):
        """
        Stage 1: Train GazePredictor supervised on real gaze heatmaps.
        All model components except GazePredictor are frozen.
        Loss: MSE between predicted and real [B, 196] heatmaps.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Freeze everything except GazePredictor
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.gaze_predictor.parameters():
            param.requires_grad = True
        # ViT needs to be active too since GazePredictor reads from its patches
        for param in self.model.image_encoder.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            list(self.model.gaze_predictor.parameters()) +
            list(self.model.image_encoder.parameters()),
            lr=self.gaze_pred_lr,
            weight_decay=self.config['training']['weight_decay']
        )

        print("\n" + "="*80)
        print("STAGE 1 — GazePredictor Training")
        print("="*80)
        print(f"Epochs: {self.gaze_pred_epochs}  |  LR: {self.gaze_pred_lr}")

        best_gaze_loss = float('inf')

        for epoch in range(self.gaze_pred_epochs):
            # ── Train ──
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1} [Train]")

            for batch in pbar:
                images, gaze_heatmaps, _, _, _, _ = self._get_batch_inputs(batch)

                optimizer.zero_grad()
                _, gaze_loss = self.model.forward_gaze_predictor(images, gaze_heatmaps)
                gaze_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += gaze_loss.item()
                pbar.set_postfix({'gaze_loss': f'{gaze_loss.item():.4f}'})
                wandb.log({"stage1/batch_gaze_loss": gaze_loss.item()})

            train_gaze_loss = total_loss / len(train_loader)

            # ── Validate ──
            self.model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images, gaze_heatmaps, _, _, _, _ = self._get_batch_inputs(batch)
                    _, gaze_loss = self.model.forward_gaze_predictor(images, gaze_heatmaps)
                    val_total += gaze_loss.item()
            val_gaze_loss = val_total / len(val_loader)

            print(f"  Epoch {epoch+1}/{self.gaze_pred_epochs} — "
                  f"Train Gaze Loss: {train_gaze_loss:.4f}  |  Val Gaze Loss: {val_gaze_loss:.4f}")

            wandb.log({
                "stage1/epoch":           epoch + 1,
                "stage1/train_gaze_loss": train_gaze_loss,
                "stage1/val_gaze_loss":   val_gaze_loss,
            })

            if val_gaze_loss < best_gaze_loss:
                best_gaze_loss = val_gaze_loss
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': self.model.state_dict(),
                    'gaze_loss':        val_gaze_loss,
                }, os.path.join(save_dir, 'best_gaze_predictor.pth'))
                print(f"  ✓ Best GazePredictor saved (Val MSE: {val_gaze_loss:.4f})")

        print(f"\n✓ Stage 1 complete. Best val gaze MSE: {best_gaze_loss:.4f}")
        return best_gaze_loss

    # ------------------------------------------------------------------
    # Stage 2 — MedGemma decoder + classification joint training
    # ------------------------------------------------------------------

    def train_medgemma(self, train_loader, val_loader, save_dir):
        """
        Stage 2: Joint training of MedGemma decoder + classification head.
        GazePredictor is frozen. Real gaze used during training.
        Loss: lambda_gen * gen_loss + lambda_cls * cls_loss
        Best model saved by val AUC.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Freeze GazePredictor — it was trained in Stage 1
        for param in self.model.gaze_predictor.parameters():
            param.requires_grad = False

        # Unfreeze everything else that should train
        # (MedGemma layers above freeze_layers, adapter, fusion, classifier)
        # These were already set correctly in MedGemmaModel.__init__
        # Re-enable ViT training
        for param in self.model.image_encoder.parameters():
            param.requires_grad = True

        # Separate param groups: ViT + fusion at base_lr, MedGemma layers at medgemma_lr
        vit_and_fusion_params = (
            list(self.model.image_encoder.parameters()) +
            list(self.model.gaze_encoder.parameters()) +
            list(self.model.level1_fusion.parameters()) +
            list(self.model.vit_adapter.parameters()) +
            list(self.model.classifier.parameters())
        )
        medgemma_params = [
            p for p in self.model.medgemma.parameters() if p.requires_grad
        ]

        optimizer = optim.AdamW([
            {'params': vit_and_fusion_params, 'lr': self.base_lr},
            {'params': medgemma_params,       'lr': self.medgemma_lr},
        ], weight_decay=self.config['training']['weight_decay'])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.medgemma_epochs,
            eta_min=1e-6
        )

        print("\n" + "="*80)
        print("STAGE 2 — MedGemma Decoder + Classification Training")
        print("="*80)
        print(f"Epochs: {self.medgemma_epochs}  |  ViT LR: {self.base_lr}  |  MedGemma LR: {self.medgemma_lr}")
        print(f"Loss: {self.lambda_gen} * gen_loss + {self.lambda_cls} * cls_loss")

        for epoch in range(self.medgemma_epochs):
            # ── Train ──
            self.model.train()

            total_loss     = 0.0
            total_gen_loss = 0.0
            total_cls_loss = 0.0
            all_labels     = []
            all_preds      = []
            all_probs      = []

            pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1} [Train]")

            for batch in pbar:
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
                labels, report_texts = self._get_batch_inputs(batch)

                report_ids, report_mask, report_labels = self._tokenize_reports(report_texts)

                optimizer.zero_grad()

                gen_loss, cls_logits, _ = self.model(
                    images=images,
                    report_token_ids=report_ids,
                    report_attention_mask=report_mask,
                    report_labels=report_labels,
                    gaze_heatmaps=gaze_heatmaps,
                    gaze_sequences=gaze_sequences,
                    gaze_seq_lengths=gaze_seq_lengths
                )

                cls_loss = self.cls_criterion(cls_logits, labels)
                loss     = self.lambda_gen * gen_loss + self.lambda_cls * cls_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                probs = torch.softmax(cls_logits, dim=1)[:, 1]
                preds = torch.argmax(cls_logits, dim=1)

                total_loss     += loss.item()
                total_gen_loss += gen_loss.item()
                total_cls_loss += cls_loss.item()
                all_labels     += labels.cpu().tolist()
                all_preds      += preds.cpu().tolist()
                all_probs      += probs.detach().cpu().tolist()

                running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                pbar.set_postfix({
                    'loss':     f'{loss.item():.4f}',
                    'gen':      f'{gen_loss.item():.4f}',
                    'cls':      f'{cls_loss.item():.4f}',
                    'acc':      f'{running_acc*100:.2f}%'
                })
                wandb.log({
                    "stage2/batch_loss":     loss.item(),
                    "stage2/batch_gen_loss": gen_loss.item(),
                    "stage2/batch_cls_loss": cls_loss.item(),
                })

            scheduler.step()

            train_loss     = total_loss     / len(train_loader)
            train_gen_loss = total_gen_loss / len(train_loader)
            train_cls_loss = total_cls_loss / len(train_loader)
            train_metrics  = self._compute_metrics(all_labels, all_preds, all_probs)

            # ── Validate ──
            val_loss, val_gen_loss, val_cls_loss, val_metrics = self._validate_stage2(
                val_loader, epoch
            )

            current_lr = scheduler.get_last_lr()[0]

            print(f"\nEpoch {epoch+1}/{self.medgemma_epochs} Summary:")
            print(f"  {'Metric':<12} {'Train':>10} {'Val':>10}")
            print(f"  {'-'*34}")
            print(f"  {'Total Loss':<12} {train_loss:>10.4f} {val_loss:>10.4f}")
            print(f"  {'Gen Loss':<12} {train_gen_loss:>10.4f} {val_gen_loss:>10.4f}")
            print(f"  {'Cls Loss':<12} {train_cls_loss:>10.4f} {val_cls_loss:>10.4f}")
            print(f"  {'Accuracy':<12} {train_metrics['accuracy']*100:>9.2f}% {val_metrics['accuracy']*100:>9.2f}%")
            print(f"  {'F1':<12} {train_metrics['f1']*100:>9.2f}% {val_metrics['f1']*100:>9.2f}%")
            print(f"  {'AUC':<12} {train_metrics['auc']:>10.4f} {val_metrics['auc']:>10.4f}")
            print(f"  LR: {current_lr:.2e}")

            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1  = val_metrics['f1']
                self.best_epoch   = epoch + 1
                self.best_metrics = val_metrics.copy()
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'), epoch, val_metrics
                )
                print(f"\n  ✓ New best model saved! (Val AUC: {val_metrics['auc']:.4f})")

            wandb.log({
                "stage2/epoch":           epoch + 1,
                "stage2/learning_rate":   current_lr,
                "stage2/train_loss":      train_loss,
                "stage2/train_gen_loss":  train_gen_loss,
                "stage2/train_cls_loss":  train_cls_loss,
                "stage2/train_accuracy":  train_metrics['accuracy'] * 100,
                "stage2/train_f1":        train_metrics['f1'] * 100,
                "stage2/train_auc":       train_metrics['auc'],
                "stage2/val_loss":        val_loss,
                "stage2/val_gen_loss":    val_gen_loss,
                "stage2/val_cls_loss":    val_cls_loss,
                "stage2/val_accuracy":    val_metrics['accuracy'] * 100,
                "stage2/val_f1":          val_metrics['f1'] * 100,
                "stage2/val_auc":         val_metrics['auc'],
                "stage2/val_best_auc":    self.best_val_auc,
                "stage2/overfitting_gap": (train_metrics['accuracy'] - val_metrics['accuracy']) * 100,
            })

            self.train_history.append({'epoch': epoch+1, 'loss': train_loss, **train_metrics})
            self.val_history.append({'epoch': epoch+1, 'loss': val_loss, **val_metrics})

        self.save_training_history(save_dir)
        return self.best_metrics

    # ------------------------------------------------------------------
    def _validate_stage2(self, val_loader, epoch):
        self.model.eval()

        total_loss     = 0.0
        total_gen_loss = 0.0
        total_cls_loss = 0.0
        all_labels     = []
        all_preds      = []
        all_probs      = []

        pbar = tqdm(val_loader, desc=f"Stage2 Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for batch in pbar:
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
                labels, report_texts = self._get_batch_inputs(batch)

                report_ids, report_mask, report_labels = self._tokenize_reports(report_texts)

                gen_loss, cls_logits, _ = self.model(
                    images=images,
                    report_token_ids=report_ids,
                    report_attention_mask=report_mask,
                    report_labels=report_labels,
                    gaze_heatmaps=gaze_heatmaps,
                    gaze_sequences=gaze_sequences,
                    gaze_seq_lengths=gaze_seq_lengths
                )

                cls_loss = self.cls_criterion(cls_logits, labels)
                loss     = self.lambda_gen * gen_loss + self.lambda_cls * cls_loss

                probs = torch.softmax(cls_logits, dim=1)[:, 1]
                preds = torch.argmax(cls_logits, dim=1)

                total_loss     += loss.item()
                total_gen_loss += gen_loss.item()
                total_cls_loss += cls_loss.item()
                all_labels     += labels.cpu().tolist()
                all_preds      += preds.cpu().tolist()
                all_probs      += probs.cpu().tolist()

                running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{running_acc*100:.2f}%'})

        val_loss     = total_loss     / len(val_loader)
        val_gen_loss = total_gen_loss / len(val_loader)
        val_cls_loss = total_cls_loss / len(val_loader)
        val_metrics  = self._compute_metrics(all_labels, all_preds, all_probs)

        return val_loss, val_gen_loss, val_cls_loss, val_metrics

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader, save_dir='checkpoints', finish_wandb=True):
        """
        Run both stages sequentially.
        Stage 1: GazePredictor
        Stage 2: MedGemma decoder + classification
        """
        os.makedirs(save_dir, exist_ok=True)

        stage1_dir = os.path.join(save_dir, 'stage1_gaze_predictor')
        stage2_dir = os.path.join(save_dir, 'stage2_medgemma')

        # Stage 1
        self.train_gaze_predictor(train_loader, val_loader, stage1_dir)

        # Stage 2
        best_metrics = self.train_medgemma(train_loader, val_loader, stage2_dir)

        if finish_wandb:
            wandb.finish()

        print("\n" + "="*80)
        print("MedGemma Training Complete!")
        print("="*80)
        print(f"\nBest Model — Epoch {self.best_epoch} (Stage 2):")
        print(f"  Accuracy:  {best_metrics['accuracy']*100:.2f}%")
        print(f"  F1:        {best_metrics['f1']*100:.2f}%")
        print(f"  AUC:       {best_metrics['auc']:.4f}")
        print(f"\nCheckpoints saved to: {save_dir}")

        return best_metrics

    # ------------------------------------------------------------------
    def evaluate_test(self, test_loader):
        """
        Run evaluation on the test set.
        Returns metrics + raw probabilities for DeLong's test.
        """
        self.model.eval()

        all_labels = []
        all_preds  = []
        all_probs  = []

        pbar = tqdm(test_loader, desc="Test Evaluation")

        with torch.no_grad():
            for batch in pbar:
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths, \
                labels, report_texts = self._get_batch_inputs(batch)

                report_ids, report_mask, report_labels = self._tokenize_reports(report_texts)

                _, cls_logits, _ = self.model(
                    images=images,
                    report_token_ids=report_ids,
                    report_attention_mask=report_mask,
                    report_labels=report_labels,
                    gaze_heatmaps=gaze_heatmaps,
                    gaze_sequences=gaze_sequences,
                    gaze_seq_lengths=gaze_seq_lengths
                )

                probs = torch.softmax(cls_logits, dim=1)[:, 1]
                preds = torch.argmax(cls_logits, dim=1)

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
    def save_checkpoint(self, path, epoch, val_metrics):
        torch.save({
            'epoch':                epoch,
            'modality':             self.modality,
            'model_state_dict':     self.model.state_dict(),
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
# main()
# ==============================================================================

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'

    print("Loading datasets...")
    train_dataset = MIMICEyeDataset(root_dir=processed_dir, split='train', config=config)
    val_dataset   = MIMICEyeDataset(root_dir=processed_dir, split='val',   config=config)

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

    use_medgemma   = config['model'].get('use_medgemma', False)
    modality       = config['model'].get('modality', 'image_gaze_text')
    timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(
        config['training']['checkpoint_dir'],
        f"{'medgemma' if use_medgemma else 'teacher'}_{modality}_{timestamp}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_medgemma:
        trainer = MedGemmaTrainer(config, train_dataset)
        trainer.train(train_loader, val_loader, save_dir=checkpoint_dir)
    else:
        trainer = C3NetTrainer(config, train_dataset)
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=config['training']['num_epochs'],
            save_dir=checkpoint_dir,
            save_periodic_checkpoints=True
        )


if __name__ == '__main__':
    main()