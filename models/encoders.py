import os
import torch
import torch.nn as nn
import timm  # PyTorch Image Models library
from transformers import AutoModel

from dotenv import load_dotenv
load_dotenv()


class ImageEncoder(nn.Module):
    """
    Vision Transformer (ViT) for image encoding.
    
    Architecture:
        Input image I: [B, C=3, H=224, W=224]
        
        1. Patch Embedding:
           Split into patches: [B, N=196, D=768]
           where N = (H/P) × (W/P) = (224/16) × (224/16) = 196
           
        2. Add positional embeddings:
           X = PatchEmbed(I) + PositionEmbed
           X: [B, N=196, D=768]
           
        3. Transformer layers (12 layers):
           For each layer l:
               X' = LayerNorm(X)
               X' = MultiHeadAttention(X') + X      # Residual connection
               X' = LayerNorm(X')
               X = FFN(X') + X'                     # Residual connection
               
        4. Output:
           - Patch features: [B, N=196, D=768]
           - Global feature (CLS token): [B, D=768]
    """
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, freeze_backbone=False):
        """
        Args:
            model_name: Name of ViT model from timm library
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: If True, freeze all layers except final projection
        """
        super(ImageEncoder, self).__init__()
        
        # Load pre-trained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Keep all tokens (don't pool)
        )
        
        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim  # 768 for ViT-Base
        self.num_patches = self.vit.patch_embed.num_patches  # 196 for 224/16
        
        if freeze_backbone:
            # Freeze all parameters
            for param in self.vit.parameters():
                param.requires_grad = False
        
        print(f"Initialized {model_name}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Num patches: {self.num_patches}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Frozen: {freeze_backbone}")
    
    def forward(self, images):
        """
        Forward pass through image encoder.
        
        Args:
            images: Tensor [B, C=3, H=224, W=224]
            
        Returns:
            patch_features: Tensor [B, N=196, D=768] - features for each patch
            global_feature: Tensor [B, D=768] - CLS token (global representation)
        """
        # Input shape: [B, 3, 224, 224]
        
        # Forward through ViT
        x = self.vit.forward_features(images)  # [B, N+1=197, D=768]
        # Note: +1 because includes CLS token
        
        # Split CLS token and patch tokens
        cls_token = x[:, 0, :]  # [B, D=768] - first token is CLS
        patch_features = x[:, 1:, :]  # [B, N=196, D=768] - remaining are patches
        
        return patch_features, cls_token


class GazeEncoder(nn.Module):
    """
    Dual-stream encoder for gaze data.
    
    Stream 1 (Spatial): Processes patch-aligned heatmap to extract attention patterns
    Stream 2 (Temporal): Processes fixation sequence to capture scan order
    
    Mathematical formulation:
    
    Spatial Stream:
        Input: H ∈ ℝ^(B×196) - patch-aligned gaze heatmap (pre-downsampled by GazePreprocessor)
        
        1. Reshape to patch grid:
           H_2d = Reshape(H, [B, 1, 14, 14])  # Align with ViT patch grid
           
        2. CNN feature extraction:
           F_spatial = CNN(H_2d)  # [B, C, H', W']
           
        3. Global pooling:
           f_spatial = GlobalAvgPool(F_spatial)  # [B, spatial_hidden_dim=256]
           
        4. Patch-level attention weights:
           W = Normalize(H)  # [B, 196, 1]
    
    Temporal Stream:
        Input: S ∈ ℝ^(B×T×3) - fixation sequence [(x,y,d), ...]
        
        1. Embedding:
           E = Linear(S)  # [B, T=50, embed_dim=128]
           
        2. LSTM encoding:
           H_t = LSTM(E)  # [B, T=50, hidden_dim=256]
           
        3. Final hidden state:
           f_temporal = H_t[:, -1, :]  # [B, temporal_hidden_dim=256]
    
    Combined output:
        - Spatial attention weights: [B, 196, 1]
        - Combined features: [B, 512] (concat of spatial and temporal)
    
    Note: GazePreprocessor in preprocessing.py downsamples the full [224,224] heatmap
    to patch-aligned [196] via 16×16 block averaging. GazeEncoder accepts this [196]
    input directly and reshapes to [1, 14, 14] internally for CNN processing.
    """
    
    def __init__(self, spatial_hidden_dim=256, temporal_hidden_dim=256, 
                 lstm_layers=2, dropout=0.3, image_size=224, patch_size=16):
        super(GazeEncoder, self).__init__()
        
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_patches = (image_size // patch_size) ** 2  # 196
        self.num_patches_1d = image_size // patch_size      # 14
        
        # ===== Spatial Stream: Process patch-aligned heatmap =====
        # Input: [B, 1, 14, 14] — reshaped from [B, 196] in forward()
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 14, 14]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # [B, 128, 14, 14]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                            # [B, 128, 1, 1]
        )
        
        self.spatial_fc = nn.Linear(128, spatial_hidden_dim)  # [B, 256]
        
        # ===== Temporal Stream: Process fixation sequence =====
        self.fixation_embedding = nn.Linear(3, 128)  # (x, y, duration) → 128-dim
        
        self.temporal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=temporal_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"Initialized GazeEncoder")
        print(f"  Input heatmap format: [B, 196] (patch-aligned, reshaped to [B, 1, 14, 14] internally)")
        print(f"  Spatial hidden dim: {spatial_hidden_dim}")
        print(f"  Temporal hidden dim: {temporal_hidden_dim}")
        print(f"  Num patches: {self.num_patches}")
    
    def forward(self, gaze_heatmaps, gaze_sequences, seq_lengths):
        """
        Forward pass through gaze encoder.
        
        Args:
            gaze_heatmaps: Tensor [B, 196] - patch-aligned gaze heatmap from GazePreprocessor
            gaze_sequences: Tensor [B, T=50, 3]
            seq_lengths: Tensor [B] - actual number of fixations per sample
            
        Returns:
            gaze_patch_weights: Tensor [B, 196, 1] - attention weight per patch
            gaze_features: Tensor [B, 512] - combined spatial+temporal features
        """
        batch_size = gaze_heatmaps.shape[0]
        
        # ===== Spatial Stream =====
        # Reshape [B, 196] → [B, 1, 14, 14] for CNN processing
        heatmap_2d = gaze_heatmaps.view(batch_size, 1, self.num_patches_1d, self.num_patches_1d)
        # heatmap_2d: [B, 1, 14, 14]
        
        # Extract spatial features via CNN
        spatial_features = self.spatial_cnn(heatmap_2d)          # [B, 128, 1, 1]
        spatial_features = spatial_features.view(batch_size, -1)  # [B, 128]
        spatial_features = self.spatial_fc(spatial_features)      # [B, 256]
        spatial_features = self.dropout(spatial_features)
        
        # Compute patch-level attention weights directly from [B, 196] input
        # Normalize to sum to 1 (probability distribution)
        # W_norm = W / Σ(W)
        patch_weights = gaze_heatmaps / (gaze_heatmaps.sum(dim=1, keepdim=True) + 1e-8)  # [B, 196]
        patch_weights = patch_weights.unsqueeze(-1)  # [B, 196, 1]
        
        # ===== Temporal Stream =====
        # Embed fixation sequence: [B, T=50, 3] → [B, T=50, 128]
        fixation_embeds = self.fixation_embedding(gaze_sequences)
        fixation_embeds = self.dropout(fixation_embeds)
        
        # Pack sequence (handle variable lengths)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            fixation_embeds,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward: [B, T=50, 128] → [B, T=50, 256]
        packed_output, (hidden, cell) = self.temporal_lstm(packed_sequence)
        
        # Unpack
        temporal_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )  # [B, T=50, 256]
        
        # Take final hidden state
        # hidden: [num_layers=2, B, 256] → take last layer
        temporal_features = hidden[-1]  # [B, 256]
        
        # ===== Combine streams =====
        # Concatenate spatial and temporal features
        gaze_features = torch.cat([spatial_features, temporal_features], dim=1)  # [B, 512]
        
        return patch_weights, gaze_features


class TextEncoder(nn.Module):
    """
    BioClinicalBERT encoder for radiology reports.
    
    Architecture:
        Input text tokens T: [B, max_length=128]
        
        1. Token Embedding + Positional Encoding:
           E = TokenEmbed(T) + PositionEmbed
           E: [B, 128, D=768]
           
        2. BERT Transformer layers (12 layers):
           For each layer l:
               E' = LayerNorm(E)
               E' = MultiHeadAttention(E', mask) + E
               E' = LayerNorm(E')
               E = FFN(E') + E'
               
        3. Output:
           - Token embeddings: [B, seq_len=128, D=768] - contextualized word representations
           - CLS token: [B, D=768] - global text representation
    
    BioClinicalBERT is pretrained on:
        - PubMed abstracts (biomedical literature)
        - MIMIC-III clinical notes (same domain as our data!)
    
    Used when config['model']['use_medgemma'] is False.
    """
    
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', freeze_bert=True):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze_bert: If True, freeze BERT weights initially (recommended for warm-up)
        """
        super(TextEncoder, self).__init__()
        
        # Load pretrained BioClinicalBERT
        print(f"Loading BERT model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get model dimensions
        self.hidden_size = self.bert.config.hidden_size  # 768
        self.num_layers = self.bert.config.num_hidden_layers  # 12
        self.max_position_embeddings = self.bert.config.max_position_embeddings  # 512
        
        # Freeze BERT parameters if requested
        self.frozen = freeze_bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        print(f"Initialized TextEncoder")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Max position embeddings: {self.max_position_embeddings}")
        print(f"  Frozen: {freeze_bert}")
    
    def forward(self, token_ids, attention_mask):
        """
        Forward pass through text encoder.
        
        Args:
            token_ids: Tensor [B, max_length=128] - tokenized input IDs
            attention_mask: Tensor [B, max_length=128] - 1 for real tokens, 0 for padding
            
        Returns:
            token_embeddings: Tensor [B, seq_len=128, D=768] - contextualized token representations
            cls_token: Tensor [B, D=768] - global text representation (CLS token)
        """
        # Forward through BERT
        # Input: token_ids [B, 128], attention_mask [B, 128]
        outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract outputs
        # last_hidden_state: [B, seq_len=128, hidden_size=768]
        token_embeddings = outputs.last_hidden_state  # [B, 128, 768]
        
        # CLS token is the first token
        cls_token = token_embeddings[:, 0, :]  # [B, 768]
        
        return token_embeddings, cls_token
    
    def unfreeze(self):
        """
        Unfreeze BERT parameters for fine-tuning.
        Call this after warm-up phase.
        """
        if self.frozen:
            print("Unfreezing TextEncoder (BERT) for fine-tuning...")
            for param in self.bert.parameters():
                param.requires_grad = True
            self.frozen = False
    
    def freeze(self):
        """
        Freeze BERT parameters.
        """
        if not self.frozen:
            print("Freezing TextEncoder (BERT)...")
            for param in self.bert.parameters():
                param.requires_grad = False
            self.frozen = True


class MedGemmaTextEncoder(nn.Module):
    """
    MedGemma text encoder — wraps the Gemma language model tower from MedGemma 4B.

    Used when config['model']['use_medgemma'] is True. Replaces BioClinicalBERT
    as the text encoder. Interface mirrors TextEncoder so teacher.py requires
    minimal branching.

    Key differences vs TextEncoder:
        - Hidden size: 2048 (Gemma 4B) vs 768 (BERT)
        - Vocab: ~256k SentencePiece vs ~30k WordPiece
        - No CLS token — we use mean pooling over non-padding tokens as the
          global representation, which is standard for decoder-style models
        - The MedGemma vision tower (SigLIP) is NOT loaded here — we use our
          own gaze-weighted ViT instead

    Output shapes:
        - token_embeddings: [B, seq_len, 2048]
        - pooled:           [B, 2048]
    """

    def __init__(self, model_name='google/medgemma-4b-it', freeze_layers=18):
        """
        Args:
            model_name:    HuggingFace MedGemma model identifier
            freeze_layers: Number of bottom Gemma transformer layers to freeze
                           (default 18 = bottom 75% of 24 layers)
        """
        super(MedGemmaTextEncoder, self).__init__()

        token = os.environ.get('HF_TOKEN')

        # Load only the language model component of MedGemma
        # We set is_text_only=True equivalent by loading just the LM backbone
        print(f"Loading MedGemma language model: {model_name}")
        from transformers import AutoModelForCausalLM
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
        )

        # Extract only the Gemma transformer backbone (no LM head, no vision tower)
        # MedGemma 4B structure: model.language_model.model (Gemma layers)
        self.gemma = full_model.language_model.model
        self.hidden_size = self.gemma.config.hidden_size  # 2048

        # Free the rest of the full model from memory
        del full_model

        # ===== Freeze bottom N layers =====
        # Always freeze the embedding layer
        for param in self.gemma.embed_tokens.parameters():
            param.requires_grad = False

        total_layers = len(self.gemma.layers)
        for i, layer in enumerate(self.gemma.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # Always train final layer norm
        for param in self.gemma.norm.parameters():
            param.requires_grad = True

        frozen_count    = freeze_layers
        trainable_count = total_layers - freeze_layers
        print(f"Initialized MedGemmaTextEncoder")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Gemma layers: {total_layers} total | {frozen_count} frozen | {trainable_count} trainable")

    def forward(self, token_ids, attention_mask):
        """
        Forward pass through Gemma text encoder.

        Args:
            token_ids:      Tensor [B, max_length=256] - tokenized input IDs
            attention_mask: Tensor [B, max_length=256] - 1 for real tokens, 0 for padding

        Returns:
            token_embeddings: Tensor [B, seq_len=256, 2048]
            pooled:           Tensor [B, 2048] - mean-pooled over non-padding tokens
        """
        outputs = self.gemma(
            input_ids=token_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        token_embeddings = outputs.last_hidden_state  # [B, seq_len, 2048]

        # Mean pooling over non-padding tokens
        # pooled = Σ(token_embeddings * mask) / Σ(mask)
        mask_expanded = attention_mask.unsqueeze(-1).float()              # [B, seq_len, 1]
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)   # [B, 2048]
        sum_mask       = mask_expanded.sum(dim=1).clamp(min=1e-9)        # [B, 1]
        pooled         = sum_embeddings / sum_mask                        # [B, 2048]

        return token_embeddings, pooled


class ViTToMedGemmaAdapter(nn.Module):
    """
    Projects ViT output features (768-dim) into MedGemma's embedding space (2048-dim).

    This adapter bridges the image encoder and the MedGemma decoder, allowing
    gaze-weighted ViT patch features and the CLS token to condition MedGemma
    generation without dimension mismatches.

    Used when config['model']['use_medgemma'] is True.

    Applied to:
        - image_cls:      [B, 768] → [B, 2048]
        - gaze_weighted:  [B, 768] → [B, 2048]  (output of GazeGuidedFusion)
        - patch_features: [B, 196, 768] → [B, 196, 2048]  (optional, for cross-attention)
    """

    def __init__(self, vit_dim=768, medgemma_dim=2048, dropout=0.1):
        """
        Args:
            vit_dim:       ViT output dimension (768 for ViT-Base)
            medgemma_dim:  MedGemma hidden dimension (2048 for 4B)
            dropout:       Dropout rate on projection output
        """
        super(ViTToMedGemmaAdapter, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(vit_dim, medgemma_dim),
            nn.LayerNorm(medgemma_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        print(f"Initialized ViTToMedGemmaAdapter")
        print(f"  ViT dim:      {vit_dim}")
        print(f"  MedGemma dim: {medgemma_dim}")

    def forward(self, features):
        """
        Args:
            features: Tensor [..., 768] — any shape ending in vit_dim
                      e.g. [B, 768] for CLS/pooled, or [B, 196, 768] for patches

        Returns:
            projected: Tensor [..., 2048] — same shape prefix, last dim projected
        """
        return self.projection(features)


class GazePredictor(nn.Module):
    """
    Auxiliary module that predicts gaze from image features alone.
    Used during training to teach image encoder to identify salient regions.

    Role in deployment pipeline:
        At inference, real gaze data is unavailable. GazePredictor generates a
        synthetic [196] heatmap from ViT patch features, which then conditions
        the MedGemma decoder — bridging the train/inference gap without
        requiring a separate student model.

    Mathematical formulation:
        Input: Patch features P ∈ ℝ^(B×N×D) from image encoder
        
        Process:
            1. Attention pooling:
               α = Softmax(W_q · P^T)  # [B, N]
               
            2. Weighted features:
               f = Σᵢ αᵢ · Pᵢ  # [B, D=768]
               
            3. MLP prediction:
               Ŵ = MLP(f)  # [B, N=196] - predicted patch weights
               
        Output: Predicted gaze weights Ŵ ∈ ℝ^(B×N)
        
        Loss: MSE(Ŵ, W_true) where W_true comes from actual gaze
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, num_patches=196):
        super(GazePredictor, self).__init__()
        
        self.num_patches = num_patches
        
        # MLP to predict patch-level saliency
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # [B, 768] → [B, 256]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),  # [B, 256] → [B, 128]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_patches)  # [B, 128] → [B, 196]
        )
        
        print(f"Initialized GazePredictor")
        print(f"  Input dim: {input_dim}")
        print(f"  Num patches: {num_patches}")
    
    def forward(self, patch_features):
        """
        Predict gaze from image patch features.
        
        Args:
            patch_features: Tensor [B, N=196, D=768]
            
        Returns:
            predicted_weights: Tensor [B, N=196] - predicted attention per patch
        """
        # Global average pooling over patches
        # Mean([B, 196, 768], dim=1) → [B, 768]
        global_features = patch_features.mean(dim=1)  # [B, 768]
        
        # Predict patch weights
        predicted_weights = self.predictor(global_features)  # [B, 196]
        
        # Apply softmax to ensure valid probability distribution
        # Ŵ = Softmax(logits)
        predicted_weights = torch.softmax(predicted_weights, dim=1)  # [B, 196]
        
        return predicted_weights


class ImageOnlyClassifier(nn.Module):
    """
    Student classifier that works with only image features at inference.
    
    During training, it can use:
    - Raw image CLS token [B, 768]
    - Gaze-guided fused features [B, 768] (from Level 1)
    
    At inference, it uses only image features.
    
    Architecture:
        Input: [B, 768] or [B, 1536] (if concatenating both)
        → MLP with hidden layers
        → Output: [B, num_classes=2]
    """
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=2, dropout=0.3):
        super(ImageOnlyClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # [B, 768] → [B, 512]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),  # [B, 512] → [B, 256]
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)  # [B, 256] → [B, 2]
        )
        
        print(f"Initialized ImageOnlyClassifier")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Dropout: {dropout}")
    
    def forward(self, features):
        """
        Args:
            features: [B, D] - can be image CLS, fused features, or concatenation
            
        Returns:
            logits: [B, num_classes=2] - raw scores (not probabilities)
        """
        logits = self.classifier(features)  # [B, 2]
        return logits