import torch
import torch.nn as nn
from transformers import BioGptTokenizer

from .teacher import MultimodalTeacher
from .decoder import C3NetDecoder


class C3NetTeacherDecoder(nn.Module):
    """
    Full C3-Net pipeline: frozen teacher + BioGPT decoder.
    Only used when config['model']['use_medgemma'] is False.

    The teacher is loaded from a pretrained checkpoint and frozen entirely.
    It acts as a feature extractor via teacher.extract_features(), producing
    the combined multimodal representation that conditions the BioGPT decoder.

    Combined feature dims (per MODALITY_DIMS in teacher.py):
        image_only:      768
        image_gaze:      768
        image_text:      2048
        image_gaze_text: 2048

    Only the projection layer and top BioGPT layers are trained.

    Pipeline:
        Image + Gaze + Text
               ↓
        Teacher (frozen) — loaded from checkpoint
               ↓
        combined_features [B, D]
               ↓
        C3NetDecoder (partially trainable)
               ↓
        Generated radiology report
    """

    def __init__(self, config, teacher_checkpoint_path):
        super(C3NetTeacherDecoder, self).__init__()

        self.config   = config
        self.modality = config['model'].get('modality', 'image_gaze_text')

        # ===== Load Teacher =====
        print("\n1. Loading pretrained teacher...")
        self.teacher = MultimodalTeacher(config=config)
        checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu', weights_only=False)
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        print(f"   ✓ Teacher loaded from: {teacher_checkpoint_path}")
        print(f"   ✓ Teacher checkpoint epoch: {checkpoint['epoch'] + 1}")
        if 'val_metrics' in checkpoint:
            val_acc = checkpoint['val_metrics']['accuracy'] * 100
            val_auc = checkpoint['val_metrics']['auc']
            print(f"   ✓ Checkpoint val accuracy: {val_acc:.2f}%  |  AUC: {val_auc:.4f}")

        # Freeze all teacher weights — we never update these
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        print(f"   ✓ Teacher frozen (all weights locked)")

        # ===== Load Decoder =====
        print("\n2. Loading C3NetDecoder (BioGPT)...")
        self.decoder = C3NetDecoder(config=config)

        # ===== Load BioGPT Tokenizer =====
        model_name     = config['decoder'].get('model_name', 'microsoft/biogpt')
        self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
        print(f"   ✓ Tokenizer loaded: {model_name}")

        # Print parameter summary
        teacher_params           = sum(p.numel() for p in self.teacher.parameters())
        dec_trainable, dec_total = self.decoder.get_trainable_params()
        print(f"\nParameter Summary:")
        print(f"  Teacher:  {teacher_params:,}  (all frozen)")
        print(f"  Decoder:  {dec_trainable:,} trainable / {dec_total:,} total")
        print(f"  Training: {dec_trainable:,} params")

    # ------------------------------------------------------------------
    def get_combined_features(self, images, gaze_heatmaps=None, gaze_sequences=None,
                               gaze_seq_lengths=None, text_token_ids=None,
                               text_attention_masks=None):
        """
        Extract combined multimodal features from the frozen teacher.
        Delegates entirely to teacher.extract_features() — no duplicated logic.

        Args:
            images:               [B, 3, 224, 224]  — always required
            gaze_heatmaps:        [B, 196]           — required for image_gaze, image_gaze_text
            gaze_sequences:       [B, 50, 3]         — required for image_gaze, image_gaze_text
            gaze_seq_lengths:     [B]                — required for image_gaze, image_gaze_text
            text_token_ids:       [B, seq_len]       — required for image_text, image_gaze_text
            text_attention_masks: [B, seq_len]       — required for image_text, image_gaze_text

        Returns:
            combined_features: [B, D] — D per MODALITY_DIMS in teacher.py
        """
        with torch.no_grad():
            combined_features, _ = self.teacher.extract_features(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks
            )

        return combined_features

    # ------------------------------------------------------------------
    def forward(self, images, report_token_ids, report_attention_mask=None,
                report_labels=None, gaze_heatmaps=None, gaze_sequences=None,
                gaze_seq_lengths=None, text_token_ids=None, text_attention_masks=None):
        """
        Forward pass with teacher forcing (used during training).

        Args:
            images:                [B, 3, 224, 224]
            report_token_ids:      [B, seq_len]  — tokenized ground truth report (input)
            report_attention_mask: [B, seq_len]  — padding mask for report tokens
            report_labels:         [B, seq_len]  — target tokens for loss (-100 to ignore)
            gaze_heatmaps:         [B, 196]      — optional, modality-dependent
            gaze_sequences:        [B, 50, 3]    — optional, modality-dependent
            gaze_seq_lengths:      [B]           — optional, modality-dependent
            text_token_ids:        [B, seq_len]  — optional, modality-dependent
            text_attention_masks:  [B, seq_len]  — optional, modality-dependent

        Returns:
            loss:   scalar generation loss
            logits: [B, seq_len+1, vocab_size]
        """

        # Extract frozen teacher features via teacher.extract_features()
        combined_features = self.get_combined_features(
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
            text_token_ids, text_attention_masks
        )
        # combined_features: [B, D]

        # Decode
        loss, logits = self.decoder(
            combined_features,
            input_ids=report_token_ids,
            attention_mask=report_attention_mask,
            labels=report_labels
        )

        return loss, logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_report(self, images, gaze_heatmaps=None, gaze_sequences=None,
                        gaze_seq_lengths=None, text_token_ids=None,
                        text_attention_masks=None):
        """
        Generate a radiology report for a batch of inputs (used during inference).

        Args:
            images:               [B, 3, 224, 224]
            gaze_heatmaps:        [B, 196]       — optional, modality-dependent
            gaze_sequences:       [B, 50, 3]     — optional, modality-dependent
            gaze_seq_lengths:     [B]            — optional, modality-dependent
            text_token_ids:       [B, seq_len]   — optional, modality-dependent
            text_attention_masks: [B, seq_len]   — optional, modality-dependent

        Returns:
            generated_texts: list of B report strings
        """

        # Extract frozen teacher features
        combined_features = self.get_combined_features(
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
            text_token_ids, text_attention_masks
        )

        # Generate text autoregressively
        generated_texts = self.decoder.generate(
            combined_features,
            tokenizer=self.tokenizer,
            config=self.config
        )

        return generated_texts

    # ------------------------------------------------------------------
    def set_decoder_train_mode(self):
        """
        Set correct train/eval modes.
        Teacher always stays eval (frozen).
        Only the decoder is set to train mode.
        """
        self.teacher.eval()   # always frozen
        self.decoder.train()

    # ------------------------------------------------------------------
    def set_eval_mode(self):
        """Set both teacher and decoder to eval mode for inference."""
        self.teacher.eval()
        self.decoder.eval()