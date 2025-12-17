import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeGuidedAttention(nn.Module):
    """
    Level 1: Gaze-Guided Visual Attention
    
    Implements the causal pathway: Image → Gaze → Attended Image Features
    
    This module uses expert gaze patterns to guide visual attention, teaching
    the model which image regions are diagnostically relevant.
    
    Mathematical Formulation:
    
    Given:
        - Image patch features: P ∈ ℝ^(B×N×D) where N=196, D=768
        - Gaze weights: W ∈ ℝ^(B×N×1) - spatial attention from expert
        - Gaze features: G ∈ ℝ^(B×512) - temporal scanning pattern
    
    Process:
        1. Simple weighted attention:
           P_weighted = P ⊙ W  (element-wise multiplication)
           F_simple = ∑ᵢ P_weighted[i]  (sum over patches)
           
        2. Learnable cross-attention:
           Q = Linear(G)  # Query from gaze pattern [B, 768]
           K = P          # Keys from image patches [B, 196, 768]
           V = P          # Values from image patches
           
           Attention scores: α = Softmax(Q @ K^T / √d_k)  [B, 196]
           Attended features: F_attn = α @ V  [B, 768]
           
        3. Combine:
           F_gaze_guided = Concat(F_simple, F_attn)  [B, 1536]
           Output = Linear(F_gaze_guided)  [B, 768]
    
    Output:
        - Gaze-weighted image features: [B, D=768]
        - Attention map: [B, N=196] for visualization
    """
    
    def __init__(self, 
                 image_dim=768,          # Image patch feature dimension
                 gaze_dim=512,           # Gaze feature dimension
                 num_heads=8,            # Multi-head attention
                 dropout=0.1):
        super(GazeGuidedAttention, self).__init__()
        
        self.image_dim = image_dim
        self.gaze_dim = gaze_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        
        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"
        
        # Transform gaze features to query space
        self.gaze_to_query = nn.Sequential(
            nn.Linear(gaze_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention projections
        self.query_proj = nn.Linear(image_dim, image_dim)
        self.key_proj = nn.Linear(image_dim, image_dim)
        self.value_proj = nn.Linear(image_dim, image_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),  # *2 because we concat simple + learned
            nn.LayerNorm(image_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        print(f"Initialized GazeGuidedAttention")
        print(f"  Image dim: {image_dim}")
        print(f"  Gaze dim: {gaze_dim}")
        print(f"  Num heads: {num_heads}")
    
    def forward(self, image_patches, gaze_weights, gaze_features):
        """
        Args:
            image_patches: [B, N=196, D=768] - patch features from ViT
            gaze_weights: [B, N=196, 1] - spatial attention from gaze heatmap
            gaze_features: [B, 512] - temporal pattern from gaze sequence
            
        Returns:
            gaze_guided_features: [B, D=768] - attended image features
            attention_map: [B, N=196] - learned attention weights (for visualization)
        """
        B, N, D = image_patches.shape
        
        # ===== Method 1: Simple Weighted Attention =====
        # Multiply each patch by its gaze weight
        # P_weighted = P ⊙ W
        weighted_patches = image_patches * gaze_weights  # [B, 196, 768]
        
        # Global average pooling: sum over patches
        # F_simple = ∑ᵢ P_weighted[i]
        simple_features = weighted_patches.sum(dim=1)  # [B, 768]
        
        # ===== Method 2: Learnable Cross-Attention =====
        # Transform gaze temporal pattern to query
        # Q = Linear(G)
        query = self.gaze_to_query(gaze_features)  # [B, 768]
        query = query.unsqueeze(1)  # [B, 1, 768] - add sequence dimension
        
        # Project for multi-head attention
        Q = self.query_proj(query)  # [B, 1, 768]
        K = self.key_proj(image_patches)  # [B, 196, 768]
        V = self.value_proj(image_patches)  # [B, 196, 768]
        
        # Reshape for multi-head attention
        # [B, 1, 768] → [B, num_heads=8, 1, head_dim=96]
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # α = Softmax(Q @ K^T / √d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, 8, 1, 196]
        attention_weights = F.softmax(scores, dim=-1)  # [B, 8, 1, 196]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # F_attn = α @ V
        attended = torch.matmul(attention_weights, V)  # [B, 8, 1, 96]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()  # [B, 1, 8, 96]
        attended = attended.view(B, 1, D)  # [B, 1, 768]
        learned_features = attended.squeeze(1)  # [B, 768]
        
        # Average attention across heads for visualization
        attention_map = attention_weights.mean(dim=1).squeeze(1)  # [B, 196]
        
        # ===== Combine Both Methods =====
        # Concatenate simple weighted + learned attention features
        combined = torch.cat([simple_features, learned_features], dim=1)  # [B, 1536]
        
        # Project to final dimension
        gaze_guided_features = self.output_proj(combined)  # [B, 768]
        
        return gaze_guided_features, attention_map


class GazeGuidedFusion(nn.Module):
    """
    Complete fusion module that wraps GazeGuidedAttention with residual connection.
    
    This ensures that even if gaze guidance fails, the model can still use 
    the original image features (residual pathway).
    """
    
    def __init__(self, image_dim=768, gaze_dim=512, num_heads=8, dropout=0.1):
        super(GazeGuidedFusion, self).__init__()
        
        self.attention = GazeGuidedAttention(
            image_dim=image_dim,
            gaze_dim=gaze_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Residual projection (in case dimensions don't match)
        self.residual_proj = nn.Identity()  # Can be Linear if needed
        
        # Layer normalization
        self.norm = nn.LayerNorm(image_dim)
    
    def forward(self, image_patches, image_cls, gaze_weights, gaze_features):
        """
        Args:
            image_patches: [B, 196, 768]
            image_cls: [B, 768] - original CLS token
            gaze_weights: [B, 196, 1]
            gaze_features: [B, 512]
            
        Returns:
            fused_features: [B, 768] - gaze-guided image features
            attention_map: [B, 196] - for visualization
        """
        # Get gaze-guided features
        gaze_guided, attention_map = self.attention(
            image_patches, 
            gaze_weights, 
            gaze_features
        )
        
        # Residual connection with original CLS token
        fused = gaze_guided + self.residual_proj(image_cls)
        
        # Layer norm
        fused = self.norm(fused)
        
        return fused, attention_map