import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import re

from ..config.config import Config


class Vocabulary:
    """Vocabulary class for managing word-to-index mappings."""
    
    def __init__(self, config: Config):
        """
        Initialize vocabulary.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Special tokens
        self.pad_token = config.data.pad_token
        self.start_token = config.data.start_token
        self.end_token = config.data.end_token
        self.unk_token = config.data.unk_token
        
        # Initialize mappings
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        
        # Special token indices
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
        self.unk_idx = 3
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [
            self.pad_token,
            self.start_token,
            self.end_token,
            self.unk_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def build_vocabulary(self, captions: List[str]) -> None:
        """
        Build vocabulary from captions.
        
        Args:
            captions: List of caption strings
        """
        print("Building vocabulary...")
        
        # Tokenize and count words
        word_counts = Counter()
        for caption in captions:
            tokens = self._tokenize(caption)
            word_counts.update(tokens)
        
        print(f"Total unique words before filtering: {len(word_counts)}")
        
        # Filter words by frequency threshold
        filtered_words = [
            word for word, count in word_counts.items()
            if count >= self.config.data.vocab_threshold
        ]
        
        # Sort by frequency (most frequent first)
        filtered_words.sort(key=lambda x: word_counts[x], reverse=True)
        
        # Limit vocabulary size
        if len(filtered_words) > self.config.data.max_vocab_size - 4:  # -4 for special tokens
            filtered_words = filtered_words[:self.config.data.max_vocab_size - 4]
        
        # Add words to vocabulary
        for word in filtered_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Final vocabulary size: {len(self.word2idx)}")
        print(f"Coverage: {self._calculate_coverage(captions):.2%}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split into words
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def _calculate_coverage(self, captions: List[str]) -> float:
        """
        Calculate vocabulary coverage over captions.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Coverage percentage
        """
        total_tokens = 0
        covered_tokens = 0
        
        for caption in captions:
            tokens = self._tokenize(caption)
            total_tokens += len(tokens)
            
            for token in tokens:
                if token in self.word2idx:
                    covered_tokens += 1
        
        return covered_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def encode_caption(self, caption: str) -> List[int]:
        """
        Encode caption to token indices.
        
        Args:
            caption: Caption string
            
        Returns:
            List of token indices
        """
        tokens = self._tokenize(caption)
        
        # Add start and end tokens
        encoded = [self.start_idx]
        
        for token in tokens:
            if token in self.word2idx:
                encoded.append(self.word2idx[token])
            else:
                encoded.append(self.unk_idx)
        
        encoded.append(self.end_idx)
        
        return encoded
    
    def decode_caption(
        self, 
        token_indices: List[int], 
        remove_special_tokens: bool = True
    ) -> str:
        """
        Decode token indices to caption string.
        
        Args:
            token_indices: List of token indices
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded caption string
        """
        tokens = []
        
        for idx in token_indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                
                # Skip special tokens if requested
                if remove_special_tokens and token in [
                    self.pad_token, self.start_token, self.end_token
                ]:
                    continue
                
                # Stop at end token
                if token == self.end_token:
                    break
                
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath: Path) -> None:
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'config': {
                'pad_token': self.pad_token,
                'start_token': self.start_token,
                'end_token': self.end_token,
                'unk_token': self.unk_token,
                'vocab_threshold': self.config.data.vocab_threshold,
                'max_vocab_size': self.config.data.max_vocab_size
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path, config: Config) -> 'Vocabulary':
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to vocabulary file
            config: Configuration object
            
        Returns:
            Loaded vocabulary object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls(config)
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {len(vocab.word2idx)}")
        
        return vocab
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2idx
    
    def get_word_frequencies(self, captions: List[str]) -> Dict[str, int]:
        """
        Get word frequencies from captions.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Dictionary of word frequencies
        """
        word_counts = Counter()
        for caption in captions:
            tokens = self._tokenize(caption)
            word_counts.update(tokens)
        
        return dict(word_counts)
    
    def get_rare_words(self, captions: List[str], threshold: int = 5) -> Set[str]:
        """
        Get words that appear less than threshold times.
        
        Args:
            captions: List of caption strings
            threshold: Frequency threshold
            
        Returns:
            Set of rare words
        """
        word_counts = self.get_word_frequencies(captions)
        return {word for word, count in word_counts.items() if count < threshold}


def build_vocabulary_from_csv(
    csv_path: Path,
    config: Config,
    caption_column: str = 'caption'
) -> Vocabulary:
    """
    Build vocabulary from CSV file containing captions.
    
    Args:
        csv_path: Path to CSV file
        config: Configuration object
        caption_column: Name of the caption column
        
    Returns:
        Built vocabulary object
    """
    print(f"Loading captions from {csv_path}")
    
    # Load captions
    df = pd.read_csv(csv_path)
    captions = df[caption_column].dropna().tolist()
    
    print(f"Loaded {len(captions)} captions")
    
    # Build vocabulary
    vocabulary = Vocabulary(config)
    vocabulary.build_vocabulary(captions)
    
    return vocabulary
