import torch
import torch.nn as nn
from .encoders import ImageEncoder, GazeEncoder, TextEncoder
from .attention import GazeGuidedFusion, TextImageAlignment


class MultimodalTeacher(nn.Module):
    """
    Teacher model for C3-Net - uses all 3 modalities for classification.
    
    Pipeline:
        Image → ViT → [patches: B,196,768] + [cls: B,768]
        Gaze  → CNN+LSTM → [weights: B,196,1] + [features: B,512]
        Level 1: GazeGuidedFusion → gaze_weighted [B,768]
        Text  → BioClinicalBERT → [embeddings: B,128,768] + [cls: B,768]
        Level 2: TextImageAlignment → aligned [B,512]
        
        Concat: [image_cls, gaze_weighted, text_cls, gaze_features, aligned]
                [768 + 768 + 768 + 512 + 512] = [B, 3328]
        
        MLP → [B, 2]
    """
    
    def __init__(self, config=None):
        super(MultimodalTeacher, self).__init__()
        
        # Hyperparameters
        num_classes = config['model'].get('num_classes', 2)    if config else 2
        freeze_bert = config['model'].get('freeze_bert', True) if config else True
        dropout     = config['model'].get('dropout', 0.3)      if config else 0.3
        
        # ===== Encoders =====
        self.image_encoder = ImageEncoder(
            model_name='vit_base_patch16_224',
            pretrained=True,
            freeze_backbone=False
        )
        
        self.gaze_encoder = GazeEncoder(
            spatial_hidden_dim=256,
            temporal_hidden_dim=256,
            lstm_layers=2,
            dropout=dropout
        )
        
        self.text_encoder = TextEncoder(
            model_name='emilyalsentzer/Bio_ClinicalBERT',
            freeze_bert=freeze_bert
        )
        
        # ===== Fusion =====
        self.level1_fusion = GazeGuidedFusion(
            image_dim=768,
            gaze_dim=512,
            num_heads=8,
            dropout=dropout
        )
        
        self.level2_fusion = TextImageAlignment(
            hidden_dim=768,
            output_dim=512,
            num_heads=8,
            dropout=dropout
        )
        
        # ===== Classifier =====
        # Concatenated feature dim: 768 + 768 + 768 + 512 + 512 = 3328
        combined_dim = 768 + 768 + 768 + 512 + 512  # 3328
        
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
        print(f"  Combined feature dim: {combined_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  BERT frozen: {freeze_bert}")
    
    def forward(self, images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks):
        """
        Args:
            images:               [B, 3, 224, 224]
            gaze_heatmaps:        [B, 224, 224]
            gaze_sequences:       [B, 50, 3]
            gaze_seq_lengths:     [B]
            text_token_ids:       [B, 128]
            text_attention_masks: [B, 128]
            
        Returns:
            logits:               [B, 2]
            attention_maps: dict with keys 'level1' [B,196], 'level2' [B,128,196]
        """
        
        # ===== Encode =====
        image_patches, image_cls = self.image_encoder(images)
        # image_patches: [B, 196, 768], image_cls: [B, 768]
        
        gaze_weights, gaze_features = self.gaze_encoder(
            gaze_heatmaps, gaze_sequences, gaze_seq_lengths
        )
        # gaze_weights: [B, 196, 1], gaze_features: [B, 512]
        
        text_embeddings, text_cls = self.text_encoder(
            text_token_ids, text_attention_masks
        )
        # text_embeddings: [B, 128, 768], text_cls: [B, 768]
        
        # ===== Level 1: Gaze-Guided Fusion =====
        gaze_weighted, attn_level1 = self.level1_fusion(
            image_patches, image_cls, gaze_weights, gaze_features
        )
        # gaze_weighted: [B, 768]
        
        # ===== Level 2: Text-Image Alignment =====
        aligned_features, attn_level2 = self.level2_fusion(
            text_embeddings, image_patches, gaze_weighted, text_attention_masks
        )
        # aligned_features: [B, 512]
        
        # ===== Combine All Features =====
        combined = torch.cat([
            image_cls,          # [B, 768] - raw image semantics
            gaze_weighted,      # [B, 768] - gaze-guided image features
            text_cls,           # [B, 768] - text semantics
            gaze_features,      # [B, 512] - temporal gaze pattern
            aligned_features    # [B, 512] - text-image aligned features
        ], dim=1)               # [B, 3328]
        
        # ===== Classify =====
        logits = self.classifier(combined)  # [B, 2]
        
        attention_maps = {
            'level1': attn_level1,  # [B, 196]
            'level2': attn_level2   # [B, 128, 196]
        }
        
        return logits, attention_maps
    
    def freeze_bert(self):
        self.text_encoder.freeze()
        print("BERT frozen")
    
    def unfreeze_bert(self):
        self.text_encoder.unfreeze()
        print("BERT unfrozen")
    
    def get_feature_vectors(self, images, gaze_heatmaps, gaze_sequences,
                            gaze_seq_lengths, text_token_ids, text_attention_masks):
        """
        Returns intermediate feature vectors for knowledge distillation.
        Specifically returns image_cls for consistency loss with student.
        """
        image_patches, image_cls = self.image_encoder(images)
        gaze_weights, gaze_features = self.gaze_encoder(
            gaze_heatmaps, gaze_sequences, gaze_seq_lengths
        )
        text_embeddings, text_cls = self.text_encoder(
            text_token_ids, text_attention_masks
        )
        gaze_weighted, attn_level1 = self.level1_fusion(
            image_patches, image_cls, gaze_weights, gaze_features
        )
        aligned_features, attn_level2 = self.level2_fusion(
            text_embeddings, image_patches, gaze_weighted, text_attention_masks
        )
        combined = torch.cat([
            image_cls, gaze_weighted, text_cls, gaze_features, aligned_features
        ], dim=1)
        logits = self.classifier(combined)
        
        return logits, image_cls  # image_cls used for consistency loss with student