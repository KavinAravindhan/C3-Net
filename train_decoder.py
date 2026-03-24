import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import json
from datetime import datetime
import wandb
from dotenv import load_dotenv
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from data.dataset import MIMICEyeDataset, collate_fn
from models.teacher_decoder import C3NetTeacherDecoder

# Load environment variables
load_dotenv()


class C3NetDecoderTrainer:
    """
    Training system for C3-Net Decoder.

    Loads the frozen teacher from a pretrained checkpoint and trains
    only the projection layer + top BioGPT layers.

    Loss:    Cross-entropy over generated tokens (teacher forcing)
    Metrics: BLEU-4, ROUGE-L, BERTScore (computed on validation set)

    Teacher forcing:
        Input  → BOS + report tokens (shifted right)
        Labels → report tokens + EOS (shifted left)
        The model learns to predict the next token at each position.
    """

    def __init__(self, config, teacher_checkpoint_path):
        self.config = config

        # Device setup
        if config['training']['device'] == 'cuda' and torch.cuda.is_available():
            gpu_id = config['training']['gpu_id']
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        print("="*80)
        print("Initializing C3-Net Decoder Training")
        print("="*80)

        # ===== Model =====
        print("\n1. Loading C3NetTeacherDecoder...")
        self.model = C3NetTeacherDecoder(
            config=config,
            teacher_checkpoint_path=teacher_checkpoint_path
        ).to(self.device)

        # Tokenizer is loaded inside C3NetTeacherDecoder
        self.tokenizer = self.model.tokenizer

        # max_length for BioGPT tokenization during training
        self.max_length = config['decoder'].get('max_length', 200)

        # ===== Optimizer =====
        # Only train decoder params — teacher is fully frozen
        decoder_lr = config['training'].get('decoder_learning_rate', 2e-5)
        trainable_params = [p for p in self.model.decoder.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=decoder_lr,
            weight_decay=config['training']['weight_decay']
        )

        num_epochs = config['training'].get('decoder_num_epochs', 20)

        # Scheduler: cosine annealing over decoder training
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )

        # ===== Metrics tracking =====
        self.best_bleu      = 0.0
        self.best_rouge     = 0.0
        self.best_epoch     = 0
        self.best_metrics   = {}
        self.train_history  = []
        self.val_history    = []

        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # ===== Wandb =====
        modality  = config['model'].get('modality', 'image_gaze_text')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(
            entity="ai4vs",
            project="c3_net",
            name=f"decoder_{modality}_{timestamp}",
            config={
                "model":               "C3NetTeacherDecoder",
                "modality":            modality,
                "decoder_lr":          decoder_lr,
                "batch_size":          config['training']['batch_size'],
                "decoder_num_epochs":  num_epochs,
                "weight_decay":        config['training']['weight_decay'],
                "biogpt_model":        config['decoder'].get('model_name', 'microsoft/biogpt'),
                "freeze_layers":       config['decoder'].get('freeze_layers', 12),
                "max_length":          self.max_length,
                "teacher_checkpoint":  teacher_checkpoint_path,
            }
        )

        print(f"\n✓ Decoder trainer initialized")
        print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"  Decoder LR:       {decoder_lr:.2e}")

    # ------------------------------------------------------------------
    def _tokenize_reports(self, report_texts):
        """
        Tokenize raw report strings with BioGPT tokenizer.

        For teacher forcing:
            input_ids:  [BOS] + tokens          (what the model receives)
            labels:     tokens + [EOS]          (what the model must predict)

        Padding is applied to max_length. Padded positions in labels
        are set to -100 so cross-entropy loss ignores them.

        Args:
            report_texts: List[str] of B raw report strings

        Returns:
            input_ids:      [B, max_length]
            attention_mask: [B, max_length]
            labels:         [B, max_length]  — -100 at pad positions
        """

        # Tokenize without special tokens first so we can manually add BOS/EOS
        encodings = self.tokenizer(
            report_texts,
            max_length=self.max_length - 1,  # leave room for BOS/EOS
            truncation=True,
            padding=False,                   # we'll pad manually below
            return_tensors=None              # returns lists for now
        )

        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        for ids in encodings['input_ids']:
            # input:  [BOS] + tokens
            inp = [bos] + ids

            # labels: tokens + [EOS]
            lbl = ids + [eos]

            # Pad both to max_length
            pad_len = self.max_length - len(inp)
            attn    = [1] * len(inp) + [0] * pad_len
            inp     = inp + [pad] * pad_len
            lbl     = lbl + [-100] * pad_len  # -100 = ignore in loss

            input_ids_list.append(inp)
            attention_mask_list.append(attn)
            labels_list.append(lbl)

        input_ids      = torch.tensor(input_ids_list,      dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)
        labels         = torch.tensor(labels_list,         dtype=torch.long).to(self.device)

        return input_ids, attention_mask, labels

    # ------------------------------------------------------------------
    def _compute_generation_metrics(self, generated_texts, reference_texts):
        """
        Compute BLEU-4, ROUGE-L, and BERTScore.

        Args:
            generated_texts: List[str] of model outputs
            reference_texts: List[str] of ground truth reports

        Returns:
            dict with keys: bleu4, rougeL, bertscore
        """

        # ===== BLEU-4 =====
        # corpus_bleu expects list of references (each a list) and hypotheses
        references  = [[ref.split()] for ref in reference_texts]
        hypotheses  = [gen.split() for gen in generated_texts]
        smoothing   = SmoothingFunction().method1  # handles short outputs gracefully
        bleu4       = corpus_bleu(references, hypotheses, smoothing_function=smoothing)

        # ===== ROUGE-L =====
        rouge_scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            score = self.rouge_scorer.score(ref, gen)
            rouge_scores.append(score['rougeL'].fmeasure)
        rougeL = np.mean(rouge_scores)

        # ===== BERTScore (using clinical model for medical text) =====
        # lang='en' with rescale_with_baseline gives more interpretable scores
        try:
            P, R, F1 = bert_score(
                generated_texts,
                reference_texts,
                lang='en',
                model_type='emilyalsentzer/Bio_ClinicalBERT',  # domain-matched scorer
                rescale_with_baseline=False,
                verbose=False
            )
            bertscore = F1.mean().item()
        except Exception as e:
            print(f"  BERTScore computation failed: {e}")
            bertscore = 0.0

        return {
            'bleu4':     bleu4,
            'rougeL':    rougeL,
            'bertscore': bertscore
        }

    # ------------------------------------------------------------------
    def _get_batch_inputs(self, batch):
        """Extract and move batch inputs to device."""
        images               = batch['images'].to(self.device)
        report_texts         = batch['report_texts']           # List[str] — stays as strings

        modality   = self.config['model'].get('modality', 'image_gaze_text')
        uses_gaze  = modality in ('image_gaze', 'image_gaze_text')
        uses_text  = modality in ('image_text', 'image_gaze_text')

        gaze_heatmaps        = batch['gaze_heatmaps'].to(self.device)        if uses_gaze else None
        gaze_sequences       = batch['gaze_sequences'].to(self.device)       if uses_gaze else None
        gaze_seq_lengths     = batch['gaze_seq_lengths']                     if uses_gaze else None
        text_token_ids       = batch['text_token_ids'].to(self.device)       if uses_text else None
        text_attention_masks = batch['text_attention_masks'].to(self.device) if uses_text else None

        return (images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks, report_texts)

    # ------------------------------------------------------------------
    def train_epoch(self, train_loader, epoch):
        """Train decoder for one epoch using teacher forcing."""
        self.model.set_decoder_train_mode()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch in pbar:
            (images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
             text_token_ids, text_attention_masks, report_texts) = self._get_batch_inputs(batch)

            # Tokenize reports with BioGPT tokenizer
            report_input_ids, report_attention_mask, report_labels = \
                self._tokenize_reports(report_texts)

            self.optimizer.zero_grad()

            loss, _ = self.model(
                images=images,
                report_token_ids=report_input_ids,
                report_attention_mask=report_attention_mask,
                report_labels=report_labels,
                gaze_heatmaps=gaze_heatmaps,
                gaze_sequences=gaze_sequences,
                gaze_seq_lengths=gaze_seq_lengths,
                text_token_ids=text_token_ids,
                text_attention_masks=text_attention_masks
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.decoder.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            wandb.log({"batch/decoder_train_loss": loss.item()})

        epoch_loss = total_loss / len(train_loader)
        return epoch_loss

    # ------------------------------------------------------------------
    def validate(self, val_loader, epoch):
        """
        Validate decoder.

        Loss:    computed with teacher forcing (fast, every batch)
        Metrics: BLEU-4, ROUGE-L, BERTScore computed on generated outputs
                 (generation is slow — runs on full val set)
        """
        self.model.set_eval_mode()

        total_loss      = 0.0
        all_generated   = []
        all_references  = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for batch in pbar:
                (images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                 text_token_ids, text_attention_masks, report_texts) = self._get_batch_inputs(batch)

                # ===== Loss (teacher forcing) =====
                report_input_ids, report_attention_mask, report_labels = \
                    self._tokenize_reports(report_texts)

                loss, _ = self.model(
                    images=images,
                    report_token_ids=report_input_ids,
                    report_attention_mask=report_attention_mask,
                    report_labels=report_labels,
                    gaze_heatmaps=gaze_heatmaps,
                    gaze_sequences=gaze_sequences,
                    gaze_seq_lengths=gaze_seq_lengths,
                    text_token_ids=text_token_ids,
                    text_attention_masks=text_attention_masks
                )
                total_loss += loss.item()

                # ===== Generate outputs for metric computation =====
                generated_texts = self.model.generate_report(
                    images=images,
                    gaze_heatmaps=gaze_heatmaps,
                    gaze_sequences=gaze_sequences,
                    gaze_seq_lengths=gaze_seq_lengths,
                    text_token_ids=text_token_ids,
                    text_attention_masks=text_attention_masks
                )

                all_generated  += generated_texts
                all_references += report_texts

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = total_loss / len(val_loader)
        metrics    = self._compute_generation_metrics(all_generated, all_references)

        return epoch_loss, metrics, all_generated, all_references

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints'):
        """Main decoder training loop."""
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*80)
        print("Starting Decoder Training")
        print("="*80)
        print(f"Total epochs:   {num_epochs}")
        print(f"Train samples:  {len(train_loader.dataset)}")
        print(f"Val samples:    {len(val_loader.dataset)}")
        print(f"Batch size:     {train_loader.batch_size}")

        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*80)

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_metrics, generated_texts, reference_texts = \
                self.validate(val_loader, epoch)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # ===== Print epoch summary =====
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  {'Metric':<12} {'Train':>10} {'Val':>10}")
            print(f"  {'-'*34}")
            print(f"  {'Loss':<12} {train_loss:>10.4f} {val_loss:>10.4f}")
            print(f"  {'BLEU-4':<12} {'':>10} {val_metrics['bleu4']:>10.4f}")
            print(f"  {'ROUGE-L':<12} {'':>10} {val_metrics['rougeL']:>10.4f}")
            print(f"  {'BERTScore':<12} {'':>10} {val_metrics['bertscore']:>10.4f}")
            print(f"  LR: {current_lr:.2e}")

            # Print a few sample generations for qualitative inspection
            print(f"\n  Sample generations (first 2):")
            for i in range(min(2, len(generated_texts))):
                print(f"\n  [Reference]: {reference_texts[i][:150]}...")
                print(f"  [Generated]: {generated_texts[i][:150]}...")

            # ===== Save best model (by BLEU-4) =====
            if val_metrics['bleu4'] > self.best_bleu:
                self.best_bleu    = val_metrics['bleu4']
                self.best_rouge   = val_metrics['rougeL']
                self.best_epoch   = epoch + 1
                self.best_metrics = val_metrics.copy()
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_decoder.pth'), epoch, val_metrics
                )
                print(f"\n  ✓ New best decoder saved! (BLEU-4: {val_metrics['bleu4']:.4f})")

            # ===== Wandb logging =====
            wandb.log({
                "epoch":          epoch + 1,
                "learning_rate":  current_lr,

                "train/loss":     train_loss,

                "val/loss":       val_loss,
                "val/bleu4":      val_metrics['bleu4'],
                "val/rougeL":     val_metrics['rougeL'],
                "val/bertscore":  val_metrics['bertscore'],

                "val/best_bleu4":     self.best_bleu,
                "val/best_rougeL":    self.best_rouge,
            })

            # History
            self.train_history.append({'epoch': epoch+1, 'loss': train_loss})
            self.val_history.append({'epoch': epoch+1, 'loss': val_loss, **val_metrics})

            # Checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'decoder_epoch_{epoch+1}.pth'),
                    epoch, val_metrics
                )

        # ===== Final summary =====
        self.save_training_history(save_dir)
        wandb.finish()

        print("\n" + "="*80)
        print("Decoder Training Complete!")
        print("="*80)
        print(f"\nBest Decoder — Epoch {self.best_epoch}:")
        print(f"  BLEU-4:    {self.best_metrics['bleu4']:.4f}")
        print(f"  ROUGE-L:   {self.best_metrics['rougeL']:.4f}")
        print(f"  BERTScore: {self.best_metrics['bertscore']:.4f}")
        print(f"\nCheckpoints saved to: {save_dir}")
        print("="*80)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path, epoch, val_metrics):
        """Save decoder checkpoint. Teacher weights are not saved — load separately."""
        torch.save({
            'epoch':                epoch,
            'modality':             self.config['model'].get('modality', 'image_gaze_text'),
            'decoder_state_dict':   self.model.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics':          val_metrics,
        }, path)

    # ------------------------------------------------------------------
    def save_training_history(self, save_dir):
        """Save full training history to JSON."""
        history = {
            'modality':     self.config['model'].get('modality', 'image_gaze_text'),
            'train':        self.train_history,
            'val':          self.val_history,
            'best_epoch':   self.best_epoch,
            'best_metrics': self.best_metrics,
        }
        with open(os.path.join(save_dir, 'decoder_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


# ==============================================================================
def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Path to the best teacher checkpoint (epoch 10 from ablations)
    # teacher_checkpoint_path = '/media/16TB_Storage/kavin/models/c3-net/teacher_20260224_004053/best_model.pth'
    teacher_checkpoint_path = '/media/16TB_Storage/kavin/models/c3-net/hparam_search/best_image_gaze_text.pth'

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

    trainer = C3NetDecoderTrainer(config, teacher_checkpoint_path)

    # Checkpoint dir
    modality       = config['model'].get('modality', 'image_gaze_text')
    timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(
        config['training']['checkpoint_dir'], f'decoder_{modality}_{timestamp}'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training'].get('decoder_num_epochs', 20),
        save_dir=checkpoint_dir
    )


if __name__ == '__main__':
    main()