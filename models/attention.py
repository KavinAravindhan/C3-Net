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


class TextImageAlignment(nn.Module):
    """
    Level 2: Text-Image Cross-Attention Alignment
    
    Implements the causal pathway: Text + Gaze-Weighted Image → Aligned Features
    
    This module learns which diagnostic terms in radiology reports correspond to 
    which visual regions in the image. For example:
    - "consolidation" → dense regions in lung fields
    - "pleural effusion" → fluid at lung bases
    - "cardiomegaly" → enlarged cardiac silhouette
    
    Mathematical Formulation:
    
    Given:
        - Text token embeddings: T ∈ ℝ^(B×seq_len×D) where seq_len=128, D=768
        - Gaze-weighted image features: I_gaze ∈ ℝ^(B×D=768)
        - Image patch features: P ∈ ℝ^(B×N×D) where N=196
    
    Process:
        1. Text-to-Image Cross-Attention:
           Q = Linear(T)  # Queries from text tokens [B, 128, 768]
           K = Linear(P)  # Keys from image patches [B, 196, 768]
           V = Linear(P)  # Values from image patches
           
           Attention: α = Softmax(Q @ K^T / √d_k)  [B, 128, 196]
           Text-grounded image: T_img = α @ V  [B, 128, 768]
           
        2. Aggregate text-grounded features:
           F_text_img = MeanPool(T_img, dim=1)  [B, 768]
           
        3. Combine with gaze-weighted image:
           F_combined = Concat(F_text_img, I_gaze)  [B, 1536]
           F_aligned = MLP(F_combined)  [B, 512]
    
    Output:
        - Text-aligned features: [B, 512]
        - Attention map: [B, 128, 196] showing text-image correspondences
    """
    
    def __init__(self, 
                 hidden_dim=768,
                 output_dim=512,
                 num_heads=8,
                 dropout=0.1):
        super(TextImageAlignment, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Cross-attention: Text queries attend to Image patches
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)  # Text → Query
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)    # Image → Key
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)  # Image → Value
        
        # Output projection after cross-attention
        self.attn_output = nn.Linear(hidden_dim, hidden_dim)
        
        # Combine text-grounded image + gaze-weighted image
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # [B, 1536] → [B, 768]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),  # [B, 768] → [B, 512]
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        print(f"Initialized TextImageAlignment")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num heads: {num_heads}")
    
    def forward(self, text_embeddings, image_patches, gaze_weighted_image, text_attention_mask=None):
        """
        Args:
            text_embeddings: [B, seq_len=128, D=768] - token embeddings from BERT
            image_patches: [B, N=196, D=768] - patch features from ViT
            gaze_weighted_image: [B, D=768] - gaze-guided image features from Level 1
            text_attention_mask: [B, seq_len=128] - optional mask for padding tokens
            
        Returns:
            aligned_features: [B, output_dim=512] - text-aligned multimodal features
            attention_map: [B, seq_len=128, N=196] - text-to-image attention weights
        """
        B, seq_len, D = text_embeddings.shape
        N = image_patches.shape[1]  # 196
        
        # ===== Cross-Attention: Text attends to Image =====
        # Project to Q, K, V
        Q = self.query_proj(text_embeddings)  # [B, 128, 768] - queries from text
        K = self.key_proj(image_patches)      # [B, 196, 768] - keys from image
        V = self.value_proj(image_patches)    # [B, 196, 768] - values from image
        
        # Reshape for multi-head attention
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, 8, 128, 96]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)        # [B, 8, 196, 96]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)        # [B, 8, 196, 96]
        
        # Compute attention scores
        # Q @ K^T: [B, 8, 128, 96] @ [B, 8, 96, 196] → [B, 8, 128, 196]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, 8, 128, 196]
        
        # Softmax over image patches (each text token attends to all patches)
        # No masking on attention scores - we handle padding during pooling
        attention_weights = F.softmax(scores, dim=-1)  # [B, 8, 128, 196]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # [B, 8, 128, 196] @ [B, 8, 196, 96] → [B, 8, 128, 96]
        attended = torch.matmul(attention_weights, V)  # [B, 8, 128, 96]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()  # [B, 128, 8, 96]
        attended = attended.view(B, seq_len, D)  # [B, 128, 768]
        
        # Output projection
        text_grounded_image = self.attn_output(attended)  # [B, 128, 768]
        
        # Average attention across heads for visualization
        attention_map = attention_weights.mean(dim=1)  # [B, 128, 196]
        
        # ===== Aggregate Text-Grounded Features =====
        # Mean pooling over text tokens (handling padding with mask)
        if text_attention_mask is not None:
            # Weighted average considering mask
            # text_attention_mask: [B, 128] where 1=real token, 0=padding
            mask_expanded = text_attention_mask.unsqueeze(-1).float()  # [B, 128, 1]
            sum_features = (text_grounded_image * mask_expanded).sum(dim=1)  # [B, 768]
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1] - prevent division by zero
            text_img_pooled = sum_features / sum_mask  # [B, 768]
        else:
            # Simple mean pooling
            text_img_pooled = text_grounded_image.mean(dim=1)  # [B, 768]
        
        # ===== Combine with Gaze-Weighted Image =====
        # Concatenate text-grounded image + gaze-weighted image
        combined = torch.cat([text_img_pooled, gaze_weighted_image], dim=1)  # [B, 1536]
        
        # Project to output dimension
        aligned_features = self.fusion_mlp(combined)  # [B, 512]
        
        return aligned_features, attention_map