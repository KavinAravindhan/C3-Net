import torch
import torch.nn as nn
from .encoders import ImageEncoder, GazeEncoder, TextEncoder, MedGemmaTextEncoder, ViTToMedGemmaAdapter
from .attention import GazeGuidedFusion, TextImageAlignment


# Modality → combined feature dim mapping (same for both pipelines):
#
#   image_only:      image_cls                                          = 768
#   image_gaze:      gaze_weighted                                      = 768
#   image_text:      image_cls + text_cls + aligned_features           = 768 + 768 + 512 = 2048
#   image_gaze_text: gaze_weighted + text_cls + aligned_features       = 768 + 768 + 512 = 2048
#
# Ablation design:
#   - image_only vs image_gaze: isolates gaze contribution via patch weighting only,
#     no extra gaze embedding dimensions added
#   - image_text vs image_gaze_text: isolates whether gaze-weighted or raw image
#     features anchor the text-image alignment
#   - gaze_features [B, 512] from GazeEncoder are used internally by
#     GazeGuidedFusion cross-attention but NOT concatenated into the final vector

MODALITY_DIMS = {
    'image_only':      768,
    'image_gaze':      768,
    'image_text':      2048,
    'image_gaze_text': 2048,
}


class MultimodalTeacher(nn.Module):
    """
    Teacher model for C3-Net — modality-configurable architecture.
    Supports two text encoder pipelines via use_medgemma flag:
        - use_medgemma=False: BioClinicalBERT (768-dim)
        - use_medgemma=True:  MedGemma text tower (2048-dim, projected to 768)

    Controlled by config['model']['modality']:

        image_only:
            Image → ViT → image_cls [B,768] → MLP → [B,2]

        image_gaze:
            Image → ViT → [patches: B,196,768] + [cls: B,768]
            Gaze  → GazeEncoder → [weights: B,196,1] + [features: B,512]
            Level 1: GazeGuidedFusion → gaze_weighted [B,768]
            Classifier input: gaze_weighted [B,768] → MLP → [B,2]

        image_text:
            Image → ViT → [patches: B,196,768] + [cls: B,768]
            Text  → TextEncoder → [embeddings: B,seq,768] + [cls: B,768]
            Level 2: TextImageAlignment (image_cls as visual anchor) → aligned [B,512]
            Classifier input: [image_cls, text_cls, aligned] [B,2048] → MLP → [B,2]

        image_gaze_text:
            Image → ViT → [patches: B,196,768] + [cls: B,768]
            Gaze  → GazeEncoder → [weights: B,196,1] + [features: B,512]
            Level 1: GazeGuidedFusion → gaze_weighted [B,768]
            Text  → TextEncoder → [embeddings: B,seq,768] + [cls: B,768]
            Level 2: TextImageAlignment (gaze_weighted as visual anchor) → aligned [B,512]
            Classifier input: [gaze_weighted, text_cls, aligned] [B,2048] → MLP → [B,2]
    """

    def __init__(self, config=None):
        super(MultimodalTeacher, self).__init__()

        # Hyperparameters
        num_classes       = config['model'].get('num_classes', 2)              if config else 2
        dropout           = config['model'].get('dropout', 0.3)                if config else 0.3
        self.modality     = config['model'].get('modality', 'image_gaze_text') if config else 'image_gaze_text'
        self.use_medgemma = config['model'].get('use_medgemma', False)         if config else False

        # bert_freeze_epochs is handled externally in train.py
        # Initialize BERT unfrozen; train.py will freeze/unfreeze as needed
        bert_freeze_epochs = config['model'].get('bert_freeze_epochs', 5) if config else 5
        freeze_bert_init   = bert_freeze_epochs > 0  # freeze at init if freeze_epochs > 0

        assert self.modality in MODALITY_DIMS, \
            f"Unknown modality '{self.modality}'. Choose from: {list(MODALITY_DIMS.keys())}"

        # ===== Encoders (conditionally initialized) =====

        # Image encoder always present
        self.image_encoder = ImageEncoder(
            model_name='vit_base_patch16_224',
            pretrained=True,
            freeze_backbone=False
        )

        # Gaze encoder: only for image_gaze and image_gaze_text
        if self.modality in ('image_gaze', 'image_gaze_text'):
            self.gaze_encoder = GazeEncoder(
                spatial_hidden_dim=256,
                temporal_hidden_dim=256,
                lstm_layers=2,
                dropout=dropout
            )

        # Text encoder: BioClinicalBERT or MedGemma, only for text-containing modalities
        if self.modality in ('image_text', 'image_gaze_text'):
            if self.use_medgemma:
                medgemma_name   = config['model']['medgemma']['model_name']   if config else 'google/medgemma-4b-it'
                medgemma_layers = config['model']['medgemma']['freeze_layers'] if config else 18
                self.text_encoder = MedGemmaTextEncoder(
                    model_name=medgemma_name,
                    freeze_layers=medgemma_layers
                )
                # Project MedGemma text output (2048) → 768 so TextImageAlignment
                # and MODALITY_DIMS remain consistent across both pipelines
                self.medgemma_text_proj = nn.Sequential(
                    nn.Linear(2048, 768),
                    nn.LayerNorm(768)
                )
            else:
                self.text_encoder = TextEncoder(
                    model_name='emilyalsentzer/Bio_ClinicalBERT',
                    freeze_bert=freeze_bert_init
                )

        # ===== Fusion (conditionally initialized) =====

        # Level 1 (Gaze-Guided): only for image_gaze and image_gaze_text
        if self.modality in ('image_gaze', 'image_gaze_text'):
            self.level1_fusion = GazeGuidedFusion(
                image_dim=768,
                gaze_dim=512,
                num_heads=8,
                dropout=dropout
            )

        # Level 2 (Text-Image Alignment): only for text-containing modalities
        # For image_text:      image_cls used as visual anchor
        # For image_gaze_text: gaze_weighted used as visual anchor
        if self.modality in ('image_text', 'image_gaze_text'):
            self.level2_fusion = TextImageAlignment(
                hidden_dim=768,
                output_dim=512,
                num_heads=8,
                dropout=dropout
            )

        # ===== Classifier =====
        combined_dim = MODALITY_DIMS[self.modality]

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

        print(f"Initialized MultimodalTeacher")
        print(f"  Modality: {self.modality}")
        print(f"  Pipeline: {'MedGemma' if self.use_medgemma else 'BioClinicalBERT + BioGPT'}")
        print(f"  Combined feature dim: {combined_dim}")
        print(f"  Num classes: {num_classes}")
        if self.modality in ('image_text', 'image_gaze_text') and not self.use_medgemma:
            print(f"  BERT frozen at init: {freeze_bert_init}")

    def _encode_text(self, text_token_ids, text_attention_masks):
        """
        Encode text using the active pipeline (BioClinicalBERT or MedGemma).
        Projects MedGemma output to 768-dim if use_medgemma is True.

        Returns:
            token_embeddings: [B, seq_len, 768]
            text_cls:         [B, 768]
        """
        token_embeddings, text_cls = self.text_encoder(text_token_ids, text_attention_masks)

        if self.use_medgemma:
            # MedGemma outputs 2048-dim — project to 768 for fusion compatibility
            token_embeddings = self.medgemma_text_proj(token_embeddings)  # [B, seq_len, 768]
            text_cls         = self.medgemma_text_proj(text_cls)          # [B, 768]

        return token_embeddings, text_cls

    def extract_features(self, images, gaze_heatmaps=None, gaze_sequences=None,
                         gaze_seq_lengths=None, text_token_ids=None, text_attention_masks=None):
        """
        Extract combined multimodal features before the classification head.
        Used for interpretability (t-SNE, GradCAM, probing) and the MedGemma decoder.

        Args:
            images:               [B, 3, 224, 224]  — always required
            gaze_heatmaps:        [B, 196]           — required for image_gaze, image_gaze_text
            gaze_sequences:       [B, 50, 3]         — required for image_gaze, image_gaze_text
            gaze_seq_lengths:     [B]                — required for image_gaze, image_gaze_text
            text_token_ids:       [B, seq_len]       — required for image_text, image_gaze_text
            text_attention_masks: [B, seq_len]       — required for image_text, image_gaze_text

        Returns:
            combined:       [B, D]  — D per MODALITY_DIMS (768 or 2048)
            attention_maps: dict — keys: 'level1' [B,196], 'level2' [B,seq_len,196]
        """

        attention_maps = {}

        # ===== Image Encoding (always) =====
        image_patches, image_cls = self.image_encoder(images)
        # image_patches: [B, 196, 768], image_cls: [B, 768]

        # ===== image_only =====
        if self.modality == 'image_only':
            # Pure ViT baseline — raw CLS token, no gaze, no text
            combined = image_cls  # [B, 768]

        # ===== image_gaze =====
        elif self.modality == 'image_gaze':
            gaze_weights, gaze_features = self.gaze_encoder(
                gaze_heatmaps, gaze_sequences, gaze_seq_lengths
            )
            # gaze_weights: [B, 196, 1], gaze_features: [B, 512]

            # Level 1: gaze guides patch attention
            # gaze_features used internally by GazeGuidedFusion cross-attention only
            gaze_weighted, attn_level1 = self.level1_fusion(
                image_patches, image_cls, gaze_weights, gaze_features
            )
            # gaze_weighted: [B, 768] — image re-weighted by expert gaze

            attention_maps['level1'] = attn_level1  # [B, 196]

            # Classifier input: gaze-weighted image only
            # gaze_features NOT concatenated — gaze is encoded into the image representation
            combined = gaze_weighted  # [B, 768]

        # ===== image_text =====
        elif self.modality == 'image_text':
            token_embeddings, text_cls = self._encode_text(
                text_token_ids, text_attention_masks
            )
            # token_embeddings: [B, seq_len, 768], text_cls: [B, 768]

            # Level 2: text tokens attend to image patches
            # image_cls used as visual anchor — no gaze in this modality
            aligned_features, attn_level2 = self.level2_fusion(
                token_embeddings, image_patches, image_cls, text_attention_masks
            )
            # aligned_features: [B, 512]

            attention_maps['level2'] = attn_level2  # [B, seq_len, 196]

            combined = torch.cat([
                image_cls,        # [B, 768] — raw image semantics
                text_cls,         # [B, 768] — text semantics
                aligned_features  # [B, 512] — text-image cross-modal alignment
            ], dim=1)             # [B, 2048]

        # ===== image_gaze_text =====
        elif self.modality == 'image_gaze_text':
            gaze_weights, gaze_features = self.gaze_encoder(
                gaze_heatmaps, gaze_sequences, gaze_seq_lengths
            )
            # gaze_weights: [B, 196, 1], gaze_features: [B, 512]

            # Level 1: Gaze-Guided Fusion
            gaze_weighted, attn_level1 = self.level1_fusion(
                image_patches, image_cls, gaze_weights, gaze_features
            )
            # gaze_weighted: [B, 768]

            token_embeddings, text_cls = self._encode_text(
                text_token_ids, text_attention_masks
            )
            # token_embeddings: [B, seq_len, 768], text_cls: [B, 768]

            # Level 2: Text-Image Alignment
            # gaze_weighted as visual anchor — text attends to gaze-informed image
            aligned_features, attn_level2 = self.level2_fusion(
                token_embeddings, image_patches, gaze_weighted, text_attention_masks
            )
            # aligned_features: [B, 512]

            attention_maps['level1'] = attn_level1  # [B, 196]
            attention_maps['level2'] = attn_level2  # [B, seq_len, 196]

            combined = torch.cat([
                gaze_weighted,    # [B, 768] — gaze-guided image features
                text_cls,         # [B, 768] — text semantics
                aligned_features  # [B, 512] — text-image cross-modal alignment
            ], dim=1)             # [B, 2048]

        return combined, attention_maps

    def forward(self, images, gaze_heatmaps=None, gaze_sequences=None, gaze_seq_lengths=None,
                text_token_ids=None, text_attention_masks=None):
        """
        Args:
            images:               [B, 3, 224, 224]  — always required
            gaze_heatmaps:        [B, 196]           — required for image_gaze, image_gaze_text
            gaze_sequences:       [B, 50, 3]         — required for image_gaze, image_gaze_text
            gaze_seq_lengths:     [B]                — required for image_gaze, image_gaze_text
            text_token_ids:       [B, seq_len]       — required for image_text, image_gaze_text
            text_attention_masks: [B, seq_len]       — required for image_text, image_gaze_text

        Returns:
            logits:         [B, 2]
            attention_maps: dict — keys: 'level1' [B,196], 'level2' [B,seq_len,196]
        """

        combined, attention_maps = self.extract_features(
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
            text_token_ids, text_attention_masks
        )

        # ===== Classify =====
        logits = self.classifier(combined)  # [B, 2]

        return logits, attention_maps

    def freeze_bert(self):
        """Freeze text encoder — only valid for text-containing modalities."""
        if hasattr(self, 'text_encoder'):
            self.text_encoder.freeze()
            print("Text encoder frozen")
        else:
            print(f"freeze_bert() called but modality is '{self.modality}' — no text encoder present, skipping.")

    def unfreeze_bert(self):
        """Unfreeze text encoder — only valid for text-containing modalities."""
        if hasattr(self, 'text_encoder'):
            self.text_encoder.unfreeze()
            print("Text encoder unfrozen")
        else:
            print(f"unfreeze_bert() called but modality is '{self.modality}' — no text encoder present, skipping.")

    def get_feature_vectors(self, images, gaze_heatmaps=None, gaze_sequences=None,
                            gaze_seq_lengths=None, text_token_ids=None, text_attention_masks=None):
        """
        Returns combined features and image_cls for downstream use.
        image_cls returned separately for visualization and probing tasks.
        """
        combined, attention_maps = self.extract_features(
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
            text_token_ids, text_attention_masks
        )

        # image_cls extracted separately for visualization / probing
        _, image_cls = self.image_encoder(images)

        return combined, image_cls, attention_maps