import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .encoders import ImageEncoder, GazeEncoder, GazePredictor, ViTToMedGemmaAdapter
from .attention import GazeGuidedFusion


class MedGemmaModel(nn.Module):
    """
    Unified MedGemma pipeline for C3-Net.
    Only used when config['model']['use_medgemma'] is True.

    Replaces the teacher-student paradigm with a single end-to-end model.
    The deployment gap (no gaze at inference) is bridged by GazePredictor,
    which generates synthetic gaze heatmaps from image features alone.

    Training pipeline:
        Image + real gaze heatmap + transcription text
               ↓
        ViT (gaze-weighted patches via GazeGuidedFusion)
               ↓
        ViTToMedGemmaAdapter  [B, 768] → [B, 2048]
               ↓
        MedGemma decoder conditioned on projected image features
               ↓
        Generated transcription + classification logits (auxiliary)

    Inference pipeline:
        Image only
               ↓
        ViT → GazePredictor → predicted heatmap [B, 196]
               ↓
        GazeGuidedFusion → gaze_weighted [B, 768]
               ↓
        ViTToMedGemmaAdapter → [B, 2048]
               ↓
        MedGemma decoder → generated transcription

    Training strategy (two-stage):
        Stage 1: Train GazePredictor supervised on real gaze heatmaps
        Stage 2: Train MedGemma decoder conditioned on predicted gaze features
                 Classification head trained jointly as auxiliary loss

    Gaze ablation:
        config['decoder']['use_gaze_conditioning'] = False zeros out the
        gaze-weighted component before the adapter, allowing comparison
        of generation quality with vs without gaze guidance.
    """

    def __init__(self, config=None):
        super(MedGemmaModel, self).__init__()

        # Hyperparameters
        num_classes          = config['model'].get('num_classes', 2)               if config else 2
        dropout              = config['model'].get('dropout', 0.3)                 if config else 0.3
        medgemma_name        = config['model']['medgemma']['model_name']           if config else 'google/medgemma-4b-it'
        medgemma_max_length  = config['model']['medgemma']['max_length']           if config else 256
        freeze_vision_tower  = config['model']['medgemma']['freeze_vision_tower']  if config else True
        medgemma_freeze_layers = config['model']['medgemma']['freeze_layers']      if config else 18
        self.use_gaze_conditioning = config['decoder'].get('use_gaze_conditioning', True) if config else True
        self.max_length      = medgemma_max_length

        # ===== Image Encoder =====
        self.image_encoder = ImageEncoder(
            model_name='vit_base_patch16_224',
            pretrained=True,
            freeze_backbone=False
        )

        # ===== Gaze Encoder =====
        # Used during training with real gaze heatmaps
        self.gaze_encoder = GazeEncoder(
            spatial_hidden_dim=256,
            temporal_hidden_dim=256,
            lstm_layers=2,
            dropout=dropout
        )

        # ===== Gaze Predictor =====
        # Used at inference — predicts synthetic heatmap from image patches alone
        self.gaze_predictor = GazePredictor(
            input_dim=768,
            hidden_dim=256,
            num_patches=196
        )

        # ===== Gaze-Guided Fusion =====
        self.level1_fusion = GazeGuidedFusion(
            image_dim=768,
            gaze_dim=512,
            num_heads=8,
            dropout=dropout
        )

        # ===== ViT → MedGemma Adapter =====
        # Projects gaze-weighted ViT features (768) → MedGemma embedding space (2048)
        self.vit_adapter = ViTToMedGemmaAdapter(
            vit_dim=768,
            medgemma_dim=2048,
            dropout=dropout
        )

        # ===== MedGemma Decoder =====
        print(f"Loading MedGemma decoder: {medgemma_name}")
        token = os.environ.get('HF_TOKEN')
        from transformers import AutoModelForCausalLM
        self.medgemma = AutoModelForCausalLM.from_pretrained(
            medgemma_name,
            token=token,
            trust_remote_code=True,
        )

        # Freeze MedGemma's own vision tower — we use our gaze-weighted ViT instead
        if freeze_vision_tower and hasattr(self.medgemma, 'vision_tower'):
            for param in self.medgemma.vision_tower.parameters():
                param.requires_grad = False
            print(f"  MedGemma vision tower frozen")

        # Freeze embedding layer
        for param in self.medgemma.language_model.model.embed_tokens.parameters():
            param.requires_grad = False

        # Freeze bottom N transformer layers, fine-tune the rest
        total_layers = len(self.medgemma.language_model.model.layers)
        for i, layer in enumerate(self.medgemma.language_model.model.layers):
            if i < medgemma_freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # Always train final norm and LM head
        for param in self.medgemma.language_model.model.norm.parameters():
            param.requires_grad = True
        for param in self.medgemma.language_model.lm_head.parameters():
            param.requires_grad = True

        frozen_count    = medgemma_freeze_layers
        trainable_count = total_layers - medgemma_freeze_layers
        print(f"  MedGemma layers: {total_layers} total | {frozen_count} frozen | {trainable_count} trainable")

        # ===== Tokenizer =====
        self.tokenizer = AutoTokenizer.from_pretrained(
            medgemma_name,
            token=token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # ===== Auxiliary Classification Head =====
        # Operates on gaze-weighted image features [B, 768]
        # Trained jointly with generation loss
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        print(f"Initialized MedGemmaModel")
        print(f"  Gaze conditioning: {self.use_gaze_conditioning}")
        print(f"  Max generation length: {self.max_length}")

    # ------------------------------------------------------------------
    def _encode_image_with_gaze(self, images, gaze_heatmaps=None,
                                 gaze_sequences=None, gaze_seq_lengths=None,
                                 use_predicted_gaze=False):
        """
        Encode image with gaze guidance.
        Uses real gaze during training, predicted gaze at inference.

        Args:
            images:            [B, 3, 224, 224]
            gaze_heatmaps:     [B, 196]     — real gaze, required if use_predicted_gaze=False
            gaze_sequences:    [B, 50, 3]   — required if use_predicted_gaze=False
            gaze_seq_lengths:  [B]          — required if use_predicted_gaze=False
            use_predicted_gaze: bool        — if True, GazePredictor generates heatmap

        Returns:
            gaze_weighted:  [B, 768]   — gaze-guided image features
            adapted:        [B, 2048]  — projected to MedGemma embedding space
            attn_map:       [B, 196]   — attention map for visualization
        """
        image_patches, image_cls = self.image_encoder(images)
        # image_patches: [B, 196, 768], image_cls: [B, 768]

        if use_predicted_gaze:
            # Inference: generate synthetic gaze from image patches
            predicted_weights = self.gaze_predictor(image_patches)  # [B, 196]
            # Reshape to [B, 196, 1] for GazeGuidedFusion
            gaze_weights = predicted_weights.unsqueeze(-1)           # [B, 196, 1]
            # Use zero temporal features since we have no fixation sequence
            gaze_features = torch.zeros(
                images.size(0), 512, device=images.device
            )                                                         # [B, 512]
        else:
            # Training: use real gaze from GazeEncoder
            gaze_weights, gaze_features = self.gaze_encoder(
                gaze_heatmaps, gaze_sequences, gaze_seq_lengths
            )
            # gaze_weights: [B, 196, 1], gaze_features: [B, 512]

        # Gaze-guided fusion
        gaze_weighted, attn_map = self.level1_fusion(
            image_patches, image_cls, gaze_weights, gaze_features
        )
        # gaze_weighted: [B, 768]

        # Optionally zero out gaze conditioning for ablation
        if not self.use_gaze_conditioning:
            gaze_weighted = image_cls  # fall back to raw ViT CLS token

        # Project to MedGemma embedding space
        adapted = self.vit_adapter(gaze_weighted)  # [B, 2048]

        return gaze_weighted, adapted, attn_map

    # ------------------------------------------------------------------
    def forward(self, images, report_token_ids, report_attention_mask=None,
                report_labels=None, gaze_heatmaps=None, gaze_sequences=None,
                gaze_seq_lengths=None):
        """
        Forward pass with teacher forcing (used during training).

        The adapted image features are prepended as a soft prefix token
        to the MedGemma input embeddings, conditioning generation on the
        gaze-weighted visual representation.

        Args:
            images:                [B, 3, 224, 224]
            report_token_ids:      [B, seq_len]  — tokenized ground truth transcription
            report_attention_mask: [B, seq_len]  — padding mask
            report_labels:         [B, seq_len]  — target tokens (-100 to ignore)
            gaze_heatmaps:         [B, 196]      — real gaze heatmap
            gaze_sequences:        [B, 50, 3]
            gaze_seq_lengths:      [B]

        Returns:
            gen_loss:    scalar — generation cross-entropy loss
            cls_logits:  [B, 2] — auxiliary classification logits
            attn_map:    [B, 196] — gaze attention map for visualization
        """

        # Encode image with real gaze
        gaze_weighted, adapted, attn_map = self._encode_image_with_gaze(
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
            use_predicted_gaze=False
        )
        # gaze_weighted: [B, 768], adapted: [B, 2048]

        # ===== MedGemma generation loss =====
        # Prepend adapted image features as prefix token to token embeddings
        prefix       = adapted.unsqueeze(1)  # [B, 1, 2048]
        token_embeds = self.medgemma.language_model.model.embed_tokens(
            report_token_ids
        )                                    # [B, seq_len, 2048]

        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        # inputs_embeds: [B, seq_len+1, 2048]

        # Extend attention mask to cover prefix token
        if report_attention_mask is not None:
            prefix_mask           = torch.ones(
                report_attention_mask.size(0), 1, device=report_attention_mask.device
            )
            report_attention_mask = torch.cat([prefix_mask, report_attention_mask], dim=1)
            # report_attention_mask: [B, seq_len+1]

        # Extend labels to ignore prefix position
        if report_labels is not None:
            prefix_ignore  = torch.full(
                (report_labels.size(0), 1), fill_value=-100, device=report_labels.device
            )
            report_labels  = torch.cat([prefix_ignore, report_labels], dim=1)
            # report_labels: [B, seq_len+1]

        outputs = self.medgemma(
            inputs_embeds=inputs_embeds,
            attention_mask=report_attention_mask,
            labels=report_labels
        )
        gen_loss = outputs.loss  # scalar

        # ===== Auxiliary classification =====
        cls_logits = self.classifier(gaze_weighted)  # [B, 2]

        return gen_loss, cls_logits, attn_map

    # ------------------------------------------------------------------
    def forward_gaze_predictor(self, images, gaze_heatmaps):
        """
        Forward pass for Stage 1 GazePredictor training.
        Supervised on real gaze heatmaps — no MedGemma involved.

        Args:
            images:        [B, 3, 224, 224]
            gaze_heatmaps: [B, 196]  — ground truth patch-aligned heatmap

        Returns:
            predicted_weights: [B, 196]  — predicted gaze heatmap
            gaze_loss:         scalar    — MSE loss vs ground truth
        """
        image_patches, _ = self.image_encoder(images)
        # image_patches: [B, 196, 768]

        predicted_weights = self.gaze_predictor(image_patches)  # [B, 196]

        # MSE loss between predicted and real gaze heatmaps
        gaze_loss = nn.functional.mse_loss(predicted_weights, gaze_heatmaps)

        return predicted_weights, gaze_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_report(self, images):
        """
        Generate a transcription using predicted gaze (inference only).
        No real gaze required — GazePredictor bridges the deployment gap.

        Args:
            images: [B, 3, 224, 224]

        Returns:
            generated_texts: list of B decoded strings
            attn_map:        [B, 196] — predicted gaze attention map
        """
        self.eval()

        _, adapted, attn_map = self._encode_image_with_gaze(
            images, use_predicted_gaze=True
        )
        # adapted: [B, 2048]

        # BOS token as starting input
        bos_ids    = torch.full(
            (images.size(0), 1),
            fill_value=self.tokenizer.bos_token_id,
            device=images.device
        )
        bos_embeds = self.medgemma.language_model.model.embed_tokens(bos_ids)
        # bos_embeds: [B, 1, 2048]

        prefix        = adapted.unsqueeze(1)                          # [B, 1, 2048]
        inputs_embeds = torch.cat([prefix, bos_embeds], dim=1)        # [B, 2, 2048]

        output_ids = self.medgemma.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_texts = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return generated_texts, attn_map

    # ------------------------------------------------------------------
    def get_trainable_params(self):
        """Returns count of trainable vs total parameters."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total