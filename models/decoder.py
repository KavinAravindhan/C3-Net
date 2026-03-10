import torch
import torch.nn as nn
from transformers import BioGptForCausalLM, BioGptConfig


class C3NetDecoder(nn.Module):
    """
    BioGPT-based text decoder for C3-Net.
    
    Takes the teacher's combined multimodal features and generates
    a radiology report via fine-tuned BioGPT.
    
    Pipeline:
        combined_features [B, 3328]
               ↓
        Projection layer → [B, 1024]  (3328 → BioGPT hidden dim)
               ↓
        BioGPT (partially frozen) → logits [B, seq_len, vocab_size]
               ↓
        Generated report text
    
    Training:  teacher forcing — ground truth tokens shifted right as input
    Inference: autoregressive beam search generation
    """

    def __init__(self, config=None):
        super(C3NetDecoder, self).__init__()

        # Hyperparameters
        model_name    = config['decoder'].get('model_name', 'microsoft/biogpt')   if config else 'microsoft/biogpt'
        input_dim     = config['decoder'].get('input_dim', 3328)                   if config else 3328
        hidden_dim    = config['decoder'].get('hidden_dim', 1024)                  if config else 1024
        freeze_layers = config['decoder'].get('freeze_layers', 12)                 if config else 12
        dropout       = config['decoder'].get('dropout', 0.1)                      if config else 0.1

        # ===== Projection =====
        # Maps teacher combined features → BioGPT hidden dim
        # LayerNorm stabilizes the projection output before feeding into BioGPT
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # refine projection
            nn.LayerNorm(hidden_dim)
        )

        # ===== BioGPT =====
        print(f"  Loading BioGPT: {model_name}...")
        self.biogpt = BioGptForCausalLM.from_pretrained(model_name)

        # ===== Freeze bottom N transformer layers =====
        # BioGPT layers are at self.biogpt.biogpt.layers
        # Freeze embedding layer always — we don't want to disturb pretrained token embeddings
        for param in self.biogpt.biogpt.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.biogpt.biogpt.embed_positions.parameters():
            param.requires_grad = False

        total_layers = len(self.biogpt.biogpt.layers)
        for i, layer in enumerate(self.biogpt.biogpt.layers):
            if i < freeze_layers:
                # Freeze bottom layers — these capture general biomedical language
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                # Fine-tune top layers — these adapt to our radiology report style
                for param in layer.parameters():
                    param.requires_grad = True

        # Always fine-tune the final layer norm and LM head
        for param in self.biogpt.biogpt.layer_norm.parameters():
            param.requires_grad = True
        for param in self.biogpt.output_projection.parameters():
            param.requires_grad = True

        frozen_layers    = freeze_layers
        trainable_layers = total_layers - freeze_layers
        print(f"  BioGPT layers: {total_layers} total | {frozen_layers} frozen | {trainable_layers} trainable")

    # ------------------------------------------------------------------
    def forward(self, combined_features, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with teacher forcing (used during training).
        
        The projected multimodal features are prepended to the token embeddings
        as a soft conditioning prefix. BioGPT then attends over both the prefix
        and the token sequence to predict the next token at each position.
        
        Args:
            combined_features: [B, input_dim]  — teacher multimodal features
            input_ids:         [B, seq_len]    — ground truth token ids (shifted right)
            attention_mask:    [B, seq_len]    — padding mask for input tokens
            labels:            [B, seq_len]    — target token ids for loss computation
                                                 (-100 at prefix positions to ignore them)
        
        Returns:
            loss:   scalar cross-entropy loss (None if labels not provided)
            logits: [B, seq_len+1, vocab_size] — predictions over vocabulary
        """

        # ===== Project multimodal features → BioGPT hidden dim =====
        prefix = self.projection(combined_features)  # [B, 1024]
        prefix = prefix.unsqueeze(1)                 # [B, 1, 1024] — single prefix token

        # ===== Get BioGPT token embeddings =====
        token_embeddings = self.biogpt.biogpt.embed_tokens(input_ids)
        # token_embeddings: [B, seq_len, 1024]

        # ===== Prepend projected features as prefix =====
        # BioGPT will condition its generation on the multimodal prefix
        inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
        # inputs_embeds: [B, seq_len+1, 1024]

        # ===== Extend attention mask to cover the prefix token =====
        if attention_mask is not None:
            prefix_mask   = torch.ones(attention_mask.size(0), 1, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            # attention_mask: [B, seq_len+1]

        # ===== Extend labels to ignore the prefix position =====
        # -100 tells cross-entropy loss to ignore that position
        if labels is not None:
            prefix_ignore = torch.full(
                (labels.size(0), 1), fill_value=-100, device=labels.device
            )
            labels = torch.cat([prefix_ignore, labels], dim=1)
            # labels: [B, seq_len+1]

        # ===== BioGPT forward pass =====
        outputs = self.biogpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss, outputs.logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, combined_features, tokenizer, config=None):
        """
        Autoregressive generation using beam search (used during inference).
        
        Args:
            combined_features: [B, input_dim]  — teacher multimodal features
            tokenizer:         BioGPT tokenizer
            config:            full config dict for generation params
        
        Returns:
            generated_texts: list of B decoded strings
        """

        # Generation hyperparameters
        max_length         = config['decoder'].get('max_length', 200)          if config else 200
        min_length         = config['decoder'].get('min_length', 50)           if config else 50
        num_beams          = config['decoder'].get('num_beams', 4)             if config else 4
        no_repeat_ngram    = config['decoder'].get('no_repeat_ngram_size', 3)  if config else 3
        early_stopping     = config['decoder'].get('early_stopping', True)     if config else True

        # Project multimodal features → prefix
        prefix = self.projection(combined_features)  # [B, 1024]
        prefix = prefix.unsqueeze(1)                 # [B, 1, 1024]

        # BOS token as starting input
        bos_ids      = torch.full(
            (combined_features.size(0), 1),
            fill_value=tokenizer.bos_token_id,
            device=combined_features.device
        )
        bos_embeds   = self.biogpt.biogpt.embed_tokens(bos_ids)  # [B, 1, 1024]
        inputs_embeds = torch.cat([prefix, bos_embeds], dim=1)    # [B, 2, 1024]

        # Generate
        output_ids = self.biogpt.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram,
            early_stopping=early_stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode token ids → strings
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return generated_texts

    # ------------------------------------------------------------------
    def get_trainable_params(self):
        """Returns count of trainable vs total parameters."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total