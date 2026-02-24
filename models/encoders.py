import torch
import torch.nn as nn
import timm  # PyTorch Image Models library
from transformers import AutoModel


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
    
    Stream 1 (Spatial): Processes heatmap to extract attention patterns
    Stream 2 (Temporal): Processes fixation sequence to capture scan order
    
    Mathematical formulation:
    
    Spatial Stream:
        Input: H ∈ ℝ^(B×H×W) - gaze heatmap
        
        1. CNN feature extraction:
           F_spatial = CNN(H)  # [B, C, H', W']
           
        2. Global pooling:
           f_spatial = GlobalAvgPool(F_spatial)  # [B, spatial_hidden_dim=256]
           
        3. Patch-level attention weights:
           W = Reshape(H, [B, 196])  # Align with image patches
           W = Normalize(W)  # [B, 196, 1]
    
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
    """
    
    def __init__(self, spatial_hidden_dim=256, temporal_hidden_dim=256, 
                 lstm_layers=2, dropout=0.3, image_size=224, patch_size=16):
        super(GazeEncoder, self).__init__()
        
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_patches = (image_size // patch_size) ** 2  # 196
        
        # ===== Spatial Stream: Process heatmap =====
        self.spatial_cnn = nn.Sequential(
            # Input: [B, 1, 224, 224]
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # [B, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 56, 56]
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 14, 14]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 14, 14]
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
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
        
        # ===== Patch-level attention weights =====
        # Convert heatmap to patch-level weights
        self.heatmap_to_patches = nn.AdaptiveAvgPool2d((14, 14))  # Match ViT patches
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"Initialized GazeEncoder")
        print(f"  Spatial hidden dim: {spatial_hidden_dim}")
        print(f"  Temporal hidden dim: {temporal_hidden_dim}")
        print(f"  Num patches: {self.num_patches}")
    
    def forward(self, gaze_heatmaps, gaze_sequences, seq_lengths):
        """
        Forward pass through gaze encoder.
        
        Args:
            gaze_heatmaps: Tensor [B, H=224, W=224]
            gaze_sequences: Tensor [B, T=50, 3]
            seq_lengths: Tensor [B] - actual number of fixations per sample
            
        Returns:
            gaze_patch_weights: Tensor [B, 196, 1] - attention weight per patch
            gaze_features: Tensor [B, 512] - combined spatial+temporal features
        """
        batch_size = gaze_heatmaps.shape[0]
        
        # ===== Spatial Stream =====
        # Add channel dimension: [B, 224, 224] → [B, 1, 224, 224]
        heatmap_input = gaze_heatmaps.unsqueeze(1)
        
        # Extract spatial features
        spatial_features = self.spatial_cnn(heatmap_input)  # [B, 128, 1, 1]
        spatial_features = spatial_features.view(batch_size, -1)  # [B, 128]
        spatial_features = self.spatial_fc(spatial_features)  # [B, 256]
        spatial_features = self.dropout(spatial_features)
        
        # Compute patch-level attention weights
        # Downsample heatmap to match patch grid: [B, 1, 224, 224] → [B, 1, 14, 14]
        patch_heatmap = self.heatmap_to_patches(heatmap_input)  # [B, 1, 14, 14]
        
        # Flatten to patch vector: [B, 1, 14, 14] → [B, 196]
        patch_weights = patch_heatmap.view(batch_size, -1)  # [B, 196]
        
        # Normalize to sum to 1 (probability distribution)
        # W_norm = W / Σ(W)
        patch_weights = patch_weights / (patch_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Add channel dimension: [B, 196] → [B, 196, 1]
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


class GazePredictor(nn.Module):
    """
    Auxiliary module that predicts gaze from image features alone.
    Used during training to teach image encoder to identify salient regions.
    
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