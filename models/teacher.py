import torch
import torch.nn as nn
from .encoders import ImageEncoder, GazeEncoder, TextEncoder
from .attention import GazeGuidedFusion, TextImageAlignment


# Modality → combined feature dim mapping:
#   image_only:      image_cls                                          = 768
#   image_gaze:      image_cls + gaze_weighted + gaze_features         = 768 + 768 + 512 = 2048
#   image_text:      image_cls + text_cls + aligned_features           = 768 + 768 + 512 = 2048
#   image_gaze_text: image_cls + gaze_weighted + text_cls +
#                    gaze_features + aligned_features                   = 768 + 768 + 768 + 512 + 512 = 3328

MODALITY_DIMS = {
    'image_only':      768,
    'image_gaze':      2048,
    'image_text':      2048,
    'image_gaze_text': 3328,
}


class MultimodalTeacher(nn.Module):
    """
    Teacher model for C3-Net - modality-configurable architecture.
    
    Controlled by config['model']['modality']:
        image_only:
            Image → ViT → [cls: B,768] → MLP → [B,2]
        
        image_gaze:
            Image → ViT → [patches: B,196,768] + [cls: B,768]
            Gaze  → CNN+LSTM → [weights: B,196,1] + [features: B,512]
            Level 1: GazeGuidedFusion → gaze_weighted [B,768]
            Concat: [image_cls, gaze_weighted, gaze_features] = [B, 2048]
            MLP → [B,2]
        
        image_text:
            Image → ViT → [patches: B,196,768] + [cls: B,768]
            Text  → BioClinicalBERT → [embeddings: B,128,768] + [cls: B,768]
            Level 2: TextImageAlignment (using image_cls as gaze substitute) → aligned [B,512]
            Concat: [image_cls, text_cls, aligned_features] = [B, 2048]
            MLP → [B,2]
        
        image_gaze_text:
            Image → ViT → [patches: B,196,768] + [cls: B,768]
            Gaze  → CNN+LSTM → [weights: B,196,1] + [features: B,512]
            Level 1: GazeGuidedFusion → gaze_weighted [B,768]
            Text  → BioClinicalBERT → [embeddings: B,128,768] + [cls: B,768]
            Level 2: TextImageAlignment → aligned [B,512]
            Concat: [image_cls, gaze_weighted, text_cls, gaze_features, aligned] = [B, 3328]
            MLP → [B,2]
    """
    
    def __init__(self, config=None):
        super(MultimodalTeacher, self).__init__()
        
        # Hyperparameters
        num_classes = config['model'].get('num_classes', 2)         if config else 2
        dropout     = config['model'].get('dropout', 0.3)           if config else 0.3
        self.modality = config['model'].get('modality', 'image_gaze_text') if config else 'image_gaze_text'
        
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
        
        # Text encoder: only for image_text and image_gaze_text
        if self.modality in ('image_text', 'image_gaze_text'):
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
        
        # Level 2 (Text-Image Alignment): only for image_text and image_gaze_text
        # Note: for image_text, image_cls is used as substitute for gaze_weighted
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
        print(f"  Combined feature dim: {combined_dim}")
        print(f"  Num classes: {num_classes}")
        if self.modality in ('image_text', 'image_gaze_text'):
            print(f"  BERT frozen at init: {freeze_bert_init}")
    
    def forward(self, images, gaze_heatmaps=None, gaze_sequences=None, gaze_seq_lengths=None,
                text_token_ids=None, text_attention_masks=None):
        """
        Args:
            images:               [B, 3, 224, 224]  — always required
            gaze_heatmaps:        [B, 224, 224]      — required for image_gaze, image_gaze_text
            gaze_sequences:       [B, 50, 3]          — required for image_gaze, image_gaze_text
            gaze_seq_lengths:     [B]                 — required for image_gaze, image_gaze_text
            text_token_ids:       [B, 128]            — required for image_text, image_gaze_text
            text_attention_masks: [B, 128]            — required for image_text, image_gaze_text
            
        Returns:
            logits:         [B, 2]
            attention_maps: dict with present keys from 'level1' [B,196], 'level2' [B,128,196]
        """
        
        attention_maps = {}
        
        # ===== Image Encoding (always) =====
        image_patches, image_cls = self.image_encoder(images)
        # image_patches: [B, 196, 768], image_cls: [B, 768]
        
        # ===== image_only =====
        if self.modality == 'image_only':
            combined = image_cls  # [B, 768]
        
        # ===== image_gaze =====
        elif self.modality == 'image_gaze':
            gaze_weights, gaze_features = self.gaze_encoder(
                gaze_heatmaps, gaze_sequences, gaze_seq_lengths
            )
            # gaze_weights: [B, 196, 1], gaze_features: [B, 512]
            
            gaze_weighted, attn_level1 = self.level1_fusion(
                image_patches, image_cls, gaze_weights, gaze_features
            )
            # gaze_weighted: [B, 768]
            
            attention_maps['level1'] = attn_level1  # [B, 196]
            
            combined = torch.cat([
                image_cls,      # [B, 768] - raw image semantics
                gaze_weighted,  # [B, 768] - gaze-guided image features
                gaze_features   # [B, 512] - temporal gaze pattern
            ], dim=1)           # [B, 2048]
        
        # ===== image_text =====
        elif self.modality == 'image_text':
            text_embeddings, text_cls = self.text_encoder(
                text_token_ids, text_attention_masks
            )
            # text_embeddings: [B, 128, 768], text_cls: [B, 768]
            
            # Use image_cls as substitute for gaze_weighted in level2 fusion
            # since gaze is not available in this modality
            aligned_features, attn_level2 = self.level2_fusion(
                text_embeddings, image_patches, image_cls, text_attention_masks
            )
            # aligned_features: [B, 512]
            
            attention_maps['level2'] = attn_level2  # [B, 128, 196]
            
            combined = torch.cat([
                image_cls,       # [B, 768] - raw image semantics
                text_cls,        # [B, 768] - text semantics
                aligned_features # [B, 512] - text-image aligned features
            ], dim=1)            # [B, 2048]
        
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
            
            text_embeddings, text_cls = self.text_encoder(
                text_token_ids, text_attention_masks
            )
            # text_embeddings: [B, 128, 768], text_cls: [B, 768]
            
            # Level 2: Text-Image Alignment
            aligned_features, attn_level2 = self.level2_fusion(
                text_embeddings, image_patches, gaze_weighted, text_attention_masks
            )
            # aligned_features: [B, 512]
            
            attention_maps['level1'] = attn_level1  # [B, 196]
            attention_maps['level2'] = attn_level2  # [B, 128, 196]
            
            combined = torch.cat([
                image_cls,          # [B, 768] - raw image semantics
                gaze_weighted,      # [B, 768] - gaze-guided image features
                text_cls,           # [B, 768] - text semantics
                gaze_features,      # [B, 512] - temporal gaze pattern
                aligned_features    # [B, 512] - text-image aligned features
            ], dim=1)               # [B, 3328]
        
        # ===== Classify =====
        logits = self.classifier(combined)  # [B, 2]
        
        return logits, attention_maps
    
    def freeze_bert(self):
        """Freeze BERT — only valid for text-containing modalities."""
        if hasattr(self, 'text_encoder'):
            self.text_encoder.freeze()
            print("BERT frozen")
        else:
            print(f"freeze_bert() called but modality is '{self.modality}' — no text encoder present, skipping.")
    
    def unfreeze_bert(self):
        """Unfreeze BERT — only valid for text-containing modalities."""
        if hasattr(self, 'text_encoder'):
            self.text_encoder.unfreeze()
            print("BERT unfrozen")
        else:
            print(f"unfreeze_bert() called but modality is '{self.modality}' — no text encoder present, skipping.")
    
    def get_feature_vectors(self, images, gaze_heatmaps=None, gaze_sequences=None,
                            gaze_seq_lengths=None, text_token_ids=None, text_attention_masks=None):
        """
        Returns intermediate feature vectors for knowledge distillation.
        Specifically returns image_cls for consistency loss with student.
        Mirrors the forward() logic but also returns image_cls separately.
        """
        logits, attention_maps = self.forward(
            images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
            text_token_ids, text_attention_masks
        )
        
        # image_cls extracted separately for student consistency loss
        _, image_cls = self.image_encoder(images)
        
        return logits, image_cls  # image_cls used for consistency loss with student