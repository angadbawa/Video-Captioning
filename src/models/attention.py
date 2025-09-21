import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config.config import Config


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""
    
    def __init__(self, config: Config):
        """
        Initialize attention mechanism.
        
        Args:
            config: Configuration object
        """
        super(BahdanauAttention, self).__init__()
        
        self.encoder_dim = config.model.encoder_hidden_dim
        self.decoder_dim = config.model.decoder_hidden_dim
        self.attention_dim = config.model.attention_dim
        
        # Linear layers for attention computation
        self.encoder_projection = nn.Linear(self.encoder_dim, self.attention_dim)
        self.decoder_projection = nn.Linear(self.decoder_dim, self.attention_dim)
        self.attention_linear = nn.Linear(self.attention_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
            encoder_mask: Optional mask for encoder outputs [batch_size, seq_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, seq_len, encoder_dim = encoder_outputs.shape
        
        # Project encoder outputs and decoder hidden state
        encoder_proj = self.encoder_projection(encoder_outputs)  # [batch_size, seq_len, attention_dim]
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Compute attention scores
        combined = torch.tanh(encoder_proj + decoder_proj)  # [batch_size, seq_len, attention_dim]
        attention_scores = self.attention_linear(combined).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(encoder_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            encoder_outputs  # [batch_size, seq_len, encoder_dim]
        ).squeeze(1)  # [batch_size, encoder_dim]
        
        return context_vector, attention_weights


class LuongAttention(nn.Module):
    """Luong (multiplicative) attention mechanism."""
    
    def __init__(self, config: Config, score_function: str = "general"):
        """
        Initialize attention mechanism.
        
        Args:
            config: Configuration object
            score_function: Type of scoring function ('dot', 'general', 'concat')
        """
        super(LuongAttention, self).__init__()
        
        self.encoder_dim = config.model.encoder_hidden_dim
        self.decoder_dim = config.model.decoder_hidden_dim
        self.attention_dim = config.model.attention_dim
        self.score_function = score_function
        
        if score_function == "general":
            self.linear_in = nn.Linear(self.decoder_dim, self.encoder_dim, bias=False)
        elif score_function == "concat":
            self.linear_query = nn.Linear(self.decoder_dim, self.attention_dim)
            self.linear_context = nn.Linear(self.encoder_dim, self.attention_dim)
            self.linear_v = nn.Linear(self.attention_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        
    def score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention scores.
        
        Args:
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            
        Returns:
            Attention scores [batch_size, seq_len]
        """
        if self.score_function == "dot":
            # Ensure dimensions match for dot product
            if self.decoder_dim != self.encoder_dim:
                raise ValueError("For dot attention, decoder and encoder dimensions must match")
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),  # [batch_size, 1, decoder_dim]
                encoder_outputs.transpose(1, 2)  # [batch_size, encoder_dim, seq_len]
            ).squeeze(1)  # [batch_size, seq_len]
            
        elif self.score_function == "general":
            projected_hidden = self.linear_in(decoder_hidden)  # [batch_size, encoder_dim]
            scores = torch.bmm(
                projected_hidden.unsqueeze(1),  # [batch_size, 1, encoder_dim]
                encoder_outputs.transpose(1, 2)  # [batch_size, encoder_dim, seq_len]
            ).squeeze(1)  # [batch_size, seq_len]
            
        elif self.score_function == "concat":
            batch_size, seq_len, _ = encoder_outputs.shape
            
            # Project decoder hidden state
            decoder_proj = self.linear_query(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
            decoder_proj = decoder_proj.expand(-1, seq_len, -1)  # [batch_size, seq_len, attention_dim]
            
            # Project encoder outputs
            encoder_proj = self.linear_context(encoder_outputs)  # [batch_size, seq_len, attention_dim]
            
            # Compute scores
            combined = torch.tanh(decoder_proj + encoder_proj)  # [batch_size, seq_len, attention_dim]
            scores = self.linear_v(combined).squeeze(-1)  # [batch_size, seq_len]
        
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")
        
        return scores
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
            encoder_mask: Optional mask for encoder outputs [batch_size, seq_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.score(decoder_hidden, encoder_outputs)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(encoder_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            encoder_outputs  # [batch_size, seq_len, encoder_dim]
        ).squeeze(1)  # [batch_size, encoder_dim]
        
        return context_vector, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: Config, num_heads: int = 8):
        """
        Initialize multi-head attention.
        
        Args:
            config: Configuration object
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        self.encoder_dim = config.model.encoder_hidden_dim
        self.decoder_dim = config.model.decoder_hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.encoder_dim // num_heads
        
        assert self.encoder_dim % num_heads == 0, "encoder_dim must be divisible by num_heads"
        
        # Linear layers for Q, K, V
        self.query_linear = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.key_linear = nn.Linear(self.encoder_dim, self.encoder_dim)
        self.value_linear = nn.Linear(self.encoder_dim, self.encoder_dim)
        
        # Output projection
        self.output_linear = nn.Linear(self.encoder_dim, self.encoder_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
            encoder_mask: Optional mask for encoder outputs [batch_size, seq_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Compute Q, K, V
        Q = self.query_linear(decoder_hidden).unsqueeze(1)  # [batch_size, 1, encoder_dim]
        K = self.key_linear(encoder_outputs)  # [batch_size, seq_len, encoder_dim]
        V = self.value_linear(encoder_outputs)  # [batch_size, seq_len, encoder_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, 1, seq_len]
        
        # Apply mask if provided
        if encoder_mask is not None:
            mask = encoder_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, 1, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, 1, head_dim]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, 1, self.encoder_dim
        ).squeeze(1)  # [batch_size, encoder_dim]
        
        # Apply output projection
        context_vector = self.output_linear(context)
        
        # Average attention weights across heads for visualization
        avg_attention_weights = attention_weights.mean(dim=1).squeeze(1)  # [batch_size, seq_len]
        
        return context_vector, avg_attention_weights


def create_attention_mechanism(config: Config, attention_type: str = "bahdanau") -> nn.Module:
    """
    Create an attention mechanism.
    
    Args:
        config: Configuration object
        attention_type: Type of attention ('bahdanau', 'luong', 'multihead')
        
    Returns:
        Attention mechanism module
    """
    if attention_type.lower() == "bahdanau":
        return BahdanauAttention(config)
    elif attention_type.lower() == "luong":
        return LuongAttention(config)
    elif attention_type.lower() == "multihead":
        return MultiHeadAttention(config)
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")
