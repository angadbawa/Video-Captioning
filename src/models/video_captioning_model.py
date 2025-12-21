import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from ..config.config import Config
from .encoder import VideoEncoder, create_feature_extractor
from .decoder import CaptionDecoder


class VideoCaptioningModel(nn.Module):
    """Complete video captioning model with encoder-decoder architecture."""
    
    def __init__(self, config: Config, vocabulary_size: int):
        """
        Initialize video captioning model.
        
        Args:
            config: Configuration object
            vocabulary_size: Size of the vocabulary
        """
        super(VideoCaptioningModel, self).__init__()
        
        self.config = config
        self.vocabulary_size = vocabulary_size
        
        # Initialize encoder and decoder
        self.encoder = VideoEncoder(config)
        self.decoder = CaptionDecoder(config, vocabulary_size)
        
        # Optional feature extractor for end-to-end training
        self.feature_extractor = None
        if hasattr(config, 'use_feature_extractor') and config.use_feature_extractor:
            self.feature_extractor = create_feature_extractor(config)
    
    def forward(
        self,
        video_features: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            video_features: Video features [batch_size, seq_len, feature_dim]
            input_tokens: Input token sequence [batch_size, target_len]
            target_tokens: Target token sequence [batch_size, target_len]
            video_mask: Optional video mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode video features
        encoder_outputs, encoder_final_state = self.encoder(video_features, video_mask)
        
        # Create encoder mask if not provided
        if video_mask is None:
            video_mask = torch.ones(
                video_features.size(0), video_features.size(1),
                device=video_features.device
            )
        
        # Decode captions
        decoder_outputs = self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_final_state=encoder_final_state,
            target_tokens=input_tokens,
            encoder_mask=video_mask
        )
        
        return {
            'logits': decoder_outputs['logits'],
            'encoder_outputs': encoder_outputs,
            'attention_weights': decoder_outputs.get('attention_weights'),
            'target_tokens': target_tokens
        }
    
    def generate(
        self,
        video_features: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 20,
        video_mask: Optional[torch.Tensor] = None,
        method: str = 'greedy',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate captions for videos.
        
        Args:
            video_features: Video features [batch_size, seq_len, feature_dim]
            start_token_id: Start token ID
            end_token_id: End token ID
            max_length: Maximum generation length
            video_mask: Optional video mask [batch_size, seq_len]
            method: Generation method ('greedy', 'beam')
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing generated captions and metadata
        """
        # Encode video features
        encoder_outputs, encoder_final_state = self.encoder(video_features, video_mask)
        
        # Create encoder mask if not provided
        if video_mask is None:
            video_mask = torch.ones(
                video_features.size(0), video_features.size(1),
                device=video_features.device
            )
        
        if method == 'greedy':
            return self._greedy_generate(
                encoder_outputs, encoder_final_state, start_token_id,
                end_token_id, max_length, video_mask, **kwargs
            )
        elif method == 'beam':
            return self._beam_search_generate(
                encoder_outputs, encoder_final_state, start_token_id,
                end_token_id, max_length, video_mask, **kwargs
            )
        else:
            raise ValueError(f"Unsupported generation method: {method}")
    
    def _greedy_generate(
        self,
        encoder_outputs: torch.Tensor,
        encoder_final_state: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        encoder_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Greedy generation."""
        return self.decoder.generate(
            encoder_outputs=encoder_outputs,
            encoder_final_state=encoder_final_state,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            max_length=max_length,
            encoder_mask=encoder_mask,
            temperature=temperature
        )
    
    def _beam_search_generate(
        self,
        encoder_outputs: torch.Tensor,
        encoder_final_state: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        encoder_mask: torch.Tensor,
        beam_size: int = 5,
        length_penalty: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Beam search generation.
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, encoder_dim]
            encoder_final_state: Final encoder state [batch_size, encoder_dim]
            start_token_id: Start token ID
            end_token_id: End token ID
            max_length: Maximum generation length
            encoder_mask: Encoder mask [batch_size, seq_len]
            beam_size: Beam size for search
            length_penalty: Length penalty factor
            
        Returns:
            Dictionary containing generated sequences
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Expand encoder outputs for beam search
        encoder_outputs = encoder_outputs.unsqueeze(1).expand(
            -1, beam_size, -1, -1
        ).contiguous().view(batch_size * beam_size, -1, encoder_outputs.size(-1))
        
        encoder_final_state = encoder_final_state.unsqueeze(1).expand(
            -1, beam_size, -1
        ).contiguous().view(batch_size * beam_size, -1)
        
        encoder_mask = encoder_mask.unsqueeze(1).expand(
            -1, beam_size, -1
        ).contiguous().view(batch_size * beam_size, -1)
        
        sequences = torch.full(
            (batch_size * beam_size, 1), start_token_id, device=device
        )
        scores = torch.zeros(batch_size * beam_size, device=device)
        
        hidden_state = self.decoder.init_hidden_state(encoder_final_state)
        
        # Keep track of completed sequences
        completed_sequences = []
        completed_scores = []
        
        for step in range(max_length):
            input_tokens = sequences[:, -1:]
            
            logits, hidden_state, _ = self.decoder.forward_step(
                input_tokens, hidden_state, encoder_outputs, encoder_mask
            )
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            candidate_scores = scores.unsqueeze(1) + log_probs
            
            candidate_scores = candidate_scores.view(batch_size, -1)
            
            top_scores, top_indices = torch.topk(
                candidate_scores, beam_size, dim=1
            )
            
            beam_indices = top_indices // self.vocabulary_size
            token_indices = top_indices % self.vocabulary_size
            
            new_sequences = []
            new_scores = []
            new_hidden_states = []
            
            for batch_idx in range(batch_size):
                for beam_idx in range(beam_size):
                    old_beam_idx = batch_idx * beam_size + beam_indices[batch_idx, beam_idx]
                    token_idx = token_indices[batch_idx, beam_idx]
                    score = top_scores[batch_idx, beam_idx]
                    
                    new_seq = torch.cat([
                        sequences[old_beam_idx],
                        token_idx.unsqueeze(0)
                    ])
                    
                    if token_idx == end_token_id:
                        length_penalty_factor = ((len(new_seq) - 1) ** length_penalty)
                        final_score = score / length_penalty_factor
                        
                        completed_sequences.append(new_seq)
                        completed_scores.append(final_score)
                    else:
                        new_sequences.append(new_seq)
                        new_scores.append(score)
                        
                        new_h = hidden_state[0][:, old_beam_idx:old_beam_idx+1, :].clone()
                        new_c = hidden_state[1][:, old_beam_idx:old_beam_idx+1, :].clone()
                        new_hidden_states.append((new_h, new_c))
            
            if not new_sequences:
                break
            
            max_len = max(len(seq) for seq in new_sequences)
            padded_sequences = []
            for seq in new_sequences:
                if len(seq) < max_len:
                    padding = torch.full(
                        (max_len - len(seq),), start_token_id, device=device
                    )
                    padded_seq = torch.cat([seq, padding])
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            
            sequences = torch.stack(padded_sequences)
            scores = torch.tensor(new_scores, device=device)
            
            if new_hidden_states:
                h_states = torch.cat([h for h, c in new_hidden_states], dim=1)
                c_states = torch.cat([c for h, c in new_hidden_states], dim=1)
                hidden_state = (h_states, c_states)
        
        if completed_sequences:
            best_sequences = []
            for batch_idx in range(batch_size):
                batch_completed = [
                    (seq, score) for seq, score in zip(completed_sequences, completed_scores)
                ]
                if batch_completed:
                    best_seq, _ = max(batch_completed, key=lambda x: x[1])
                    best_sequences.append(best_seq)
                else:
                    best_sequences.append(sequences[batch_idx * beam_size])
        else:
            best_sequences = [sequences[i * beam_size] for i in range(batch_size)]
        
        max_len = max(len(seq) for seq in best_sequences)
        final_sequences = []
        for seq in best_sequences:
            if len(seq) < max_len:
                padding = torch.full(
                    (max_len - len(seq),), start_token_id, device=device
                )
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            final_sequences.append(padded_seq)
        
        generated_tokens = torch.stack(final_sequences)
        
        return {'generated_tokens': generated_tokens}
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
