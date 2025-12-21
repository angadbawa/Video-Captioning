import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from ..config.config import Config
from .attention import create_attention_mechanism


class CaptionDecoder(nn.Module):
    """LSTM-based decoder with attention for caption generation."""
    
    def __init__(self, config: Config, vocabulary_size: int):
        """
        Initialize caption decoder.
        
        Args:
            config: Configuration object
            vocabulary_size: Size of the vocabulary
        """
        super(CaptionDecoder, self).__init__()
        
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = config.model.embedding_dim
        self.hidden_dim = config.model.decoder_hidden_dim
        self.encoder_dim = config.model.encoder_hidden_dim
        self.num_layers = config.model.decoder_num_layers
        self.dropout = config.model.decoder_dropout
        self.use_attention = config.model.use_attention
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocabulary_size, self.embedding_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)
        
        # Attention mechanism
        if self.use_attention:
            self.attention = create_attention_mechanism(config, "bahdanau")
            lstm_input_size = self.embedding_dim + self.encoder_dim
        else:
            lstm_input_size = self.embedding_dim
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output projection layers
        if self.use_attention:
            self.context_projection = nn.Linear(
                self.encoder_dim + self.hidden_dim + self.embedding_dim,
                self.hidden_dim
            )
        
        self.output_projection = nn.Linear(self.hidden_dim, vocabulary_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
        
        if self.use_attention:
            nn.init.xavier_uniform_(self.context_projection.weight)
            nn.init.constant_(self.context_projection.bias, 0)
    
    def init_hidden_state(
        self, 
        encoder_final_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize decoder hidden state from encoder final state.
        
        Args:
            encoder_final_state: Final encoder state [batch_size, encoder_dim]
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        batch_size = encoder_final_state.size(0)
        
        # Project encoder state to decoder hidden dimension
        if self.encoder_dim != self.hidden_dim:
            projection = nn.Linear(self.encoder_dim, self.hidden_dim).to(encoder_final_state.device)
            projected_state = projection(encoder_final_state)
        else:
            projected_state = encoder_final_state
        
        hidden_state = projected_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state)
        
        return hidden_state, cell_state
    
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward step of the decoder.
        
        Args:
            input_token: Input token [batch_size, 1]
            hidden_state: Previous hidden state (h, c)
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            encoder_mask: Optional encoder mask [batch_size, seq_len]
            
        Returns:
            Tuple of (output_logits, new_hidden_state, attention_weights)
        """
        batch_size = input_token.size(0)
        
        # Embed input token
        embedded = self.embedding(input_token)  # [batch_size, 1, embedding_dim]
        embedded = self.embedding_dropout(embedded)
        
        # Prepare LSTM input
        if self.use_attention:
            current_hidden = hidden_state[0][-1]  # [batch_size, hidden_dim]
            
            # Compute attention
            context_vector, attention_weights = self.attention(
                encoder_outputs, current_hidden, encoder_mask
            )
            
            # Concatenate embedding and context
            lstm_input = torch.cat([
                embedded,
                context_vector.unsqueeze(1)
            ], dim=2)  # [batch_size, 1, embedding_dim + encoder_dim]
        else:
            lstm_input = embedded
            attention_weights = None
        
        # LSTM forward pass
        lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        
        # Prepare output projection input
        if self.use_attention:
            # Concatenate LSTM output, context, and embedding for output projection
            projection_input = torch.cat([
                lstm_output.squeeze(1),  # [batch_size, hidden_dim]
                context_vector,  # [batch_size, encoder_dim]
                embedded.squeeze(1)  # [batch_size, embedding_dim]
            ], dim=1)
            
            # Apply context projection
            projected_output = self.context_projection(projection_input)
            projected_output = torch.tanh(projected_output)
        else:
            projected_output = lstm_output.squeeze(1)
        
        output_logits = self.output_projection(projected_output)
        
        return output_logits, new_hidden_state, attention_weights
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_final_state: torch.Tensor,
        target_tokens: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            encoder_final_state: Final encoder state [batch_size, encoder_dim]
            target_tokens: Target token sequence [batch_size, target_len]
            encoder_mask: Optional encoder mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing logits and attention weights
        """
        batch_size, target_len = target_tokens.shape
        
        hidden_state = self.init_hidden_state(encoder_final_state)
        
        # Prepare outputs
        all_logits = []
        all_attention_weights = []
        
        # Teacher forcing: use target tokens as input
        for t in range(target_len):
            input_token = target_tokens[:, t:t+1]  # [batch_size, 1]
            
            logits, hidden_state, attention_weights = self.forward_step(
                input_token, hidden_state, encoder_outputs, encoder_mask
            )
            
            all_logits.append(logits)
            if attention_weights is not None:
                all_attention_weights.append(attention_weights)
        
        # Stack outputs
        output_logits = torch.stack(all_logits, dim=1)  # [batch_size, target_len, vocab_size]
        
        result = {'logits': output_logits}
        
        if all_attention_weights:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # [batch_size, target_len, seq_len]
            result['attention_weights'] = attention_weights
        
        return result
    
    def generate(
        self,
        encoder_outputs: torch.Tensor,
        encoder_final_state: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 20,
        encoder_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate captions using greedy decoding.
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            encoder_final_state: Final encoder state [batch_size, encoder_dim]
            start_token_id: Start token ID
            end_token_id: End token ID
            max_length: Maximum generation length
            encoder_mask: Optional encoder mask [batch_size, seq_len]
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing generated tokens and attention weights
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        hidden_state = self.init_hidden_state(encoder_final_state)
        
        input_token = torch.full((batch_size, 1), start_token_id, device=device)
        
        # Storage for outputs
        generated_tokens = []
        all_attention_weights = []
        
        for _ in range(max_length):
            logits, hidden_state, attention_weights = self.forward_step(
                input_token, hidden_state, encoder_outputs, encoder_mask
            )
            
            # Apply temperature and get next token
            if temperature != 1.0:
                logits = logits / temperature
            
            # Greedy selection
            next_token = torch.argmax(logits, dim=1, keepdim=True)
            
            generated_tokens.append(next_token)
            if attention_weights is not None:
                all_attention_weights.append(attention_weights)
            
            if (next_token == end_token_id).all():
                break
            
            input_token = next_token
        
        # Stack outputs
        generated_sequence = torch.cat(generated_tokens, dim=1)  # [batch_size, gen_len]
        
        result = {'generated_tokens': generated_sequence}
        
        if all_attention_weights:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # [batch_size, gen_len, seq_len]
            result['attention_weights'] = attention_weights
        
        return result
