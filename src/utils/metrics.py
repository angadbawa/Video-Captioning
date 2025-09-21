import re
from typing import List, Dict, Any
from collections import Counter
import math

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

from ..data.vocabulary import Vocabulary


class CaptionMetrics:
    """Metrics for evaluating video captions."""
    
    def __init__(self, vocabulary: Vocabulary):
        """
        Initialize metrics.
        
        Args:
            vocabulary: Vocabulary object for text processing
        """
        self.vocabulary = vocabulary
        
        if NLTK_AVAILABLE:
            self.smoothing_function = SmoothingFunction().method4
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
    
    def compute_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self._compute_basic_metrics(predictions, references))
        
        # BLEU scores
        if NLTK_AVAILABLE:
            metrics.update(self._compute_bleu_scores(predictions, references))
            
            # METEOR score
            try:
                metrics['meteor'] = self._compute_meteor_score(predictions, references)
            except:
                pass
        
        # ROUGE scores
        if ROUGE_AVAILABLE:
            metrics.update(self._compute_rouge_scores(predictions, references))
        
        # CIDEr score (simplified implementation)
        metrics['cider'] = self._compute_cider_score(predictions, references)
        
        return metrics
    
    def _compute_basic_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute basic metrics like length and vocabulary overlap."""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        # Average lengths
        avg_pred_length = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
        avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
        
        # Vocabulary overlap
        pred_vocab = set()
        ref_vocab = set()
        
        for pred in predictions:
            pred_vocab.update(pred.lower().split())
        
        for ref in references:
            ref_vocab.update(ref.lower().split())
        
        vocab_overlap = len(pred_vocab & ref_vocab) / len(pred_vocab | ref_vocab) if pred_vocab | ref_vocab else 0
        
        return {
            'avg_pred_length': avg_pred_length,
            'avg_ref_length': avg_ref_length,
            'vocab_overlap': vocab_overlap
        }
    
    def _compute_bleu_scores(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores."""
        bleu_scores = {'bleu_1': 0, 'bleu_2': 0, 'bleu_3': 0, 'bleu_4': 0}
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]  # BLEU expects list of reference lists
            
            # Compute BLEU scores for different n-grams
            for n in range(1, 5):
                weights = [1/n] * n + [0] * (4-n)
                try:
                    score = sentence_bleu(
                        ref_tokens, pred_tokens, 
                        weights=weights, 
                        smoothing_function=self.smoothing_function
                    )
                    bleu_scores[f'bleu_{n}'] += score
                except:
                    pass
        
        # Average scores
        num_samples = len(predictions)
        for key in bleu_scores:
            bleu_scores[key] /= num_samples
        
        return bleu_scores
    
    def _compute_meteor_score(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """Compute METEOR score."""
        total_score = 0
        
        for pred, ref in zip(predictions, references):
            try:
                score = meteor_score([ref.lower().split()], pred.lower().split())
                total_score += score
            except:
                pass
        
        return total_score / len(predictions) if predictions else 0
    
    def _compute_rouge_scores(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        
        # Average scores
        num_samples = len(predictions)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples
        
        return rouge_scores
    
    def _compute_cider_score(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """Compute simplified CIDEr score."""
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Get n-grams from tokens."""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i+n]))
            return Counter(ngrams)
        
        def compute_tf_idf(ngrams: Counter, doc_freq: Dict[str, int], num_docs: int) -> Dict[str, float]:
            """Compute TF-IDF scores."""
            tf_idf = {}
            for ngram, count in ngrams.items():
                tf = count / sum(ngrams.values()) if ngrams.values() else 0
                idf = math.log(num_docs / doc_freq.get(ngram, 1))
                tf_idf[ngram] = tf * idf
            return tf_idf
        
        # Collect all n-grams and compute document frequencies
        all_ngrams = set()
        doc_frequencies = {}
        
        # Process references and predictions
        all_texts = predictions + references
        for text in all_texts:
            tokens = text.lower().split()
            for n in range(1, 5):  # 1-4 grams
                ngrams = get_ngrams(tokens, n)
                for ngram in ngrams:
                    all_ngrams.add(ngram)
                    doc_frequencies[ngram] = doc_frequencies.get(ngram, 0) + 1
        
        # Compute CIDEr scores
        total_score = 0
        num_docs = len(all_texts)
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            score = 0
            for n in range(1, 5):
                pred_ngrams = get_ngrams(pred_tokens, n)
                ref_ngrams = get_ngrams(ref_tokens, n)
                
                pred_tfidf = compute_tf_idf(pred_ngrams, doc_frequencies, num_docs)
                ref_tfidf = compute_tf_idf(ref_ngrams, doc_frequencies, num_docs)
                
                # Compute cosine similarity
                dot_product = sum(
                    pred_tfidf.get(ngram, 0) * ref_tfidf.get(ngram, 0)
                    for ngram in all_ngrams
                )
                
                pred_norm = math.sqrt(sum(v**2 for v in pred_tfidf.values()))
                ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
                
                if pred_norm > 0 and ref_norm > 0:
                    score += dot_product / (pred_norm * ref_norm)
            
            total_score += score / 4  # Average over n-gram orders
        
        return total_score / len(predictions) if predictions else 0
    
    def compute_diversity_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute diversity metrics for generated captions.
        
        Args:
            predictions: List of predicted captions
            
        Returns:
            Dictionary of diversity metrics
        """
        if not predictions:
            return {}
        
        # Collect all tokens
        all_tokens = []
        for pred in predictions:
            all_tokens.extend(pred.lower().split())
        
        # Unique tokens
        unique_tokens = set(all_tokens)
        vocab_size = len(unique_tokens)
        total_tokens = len(all_tokens)
        
        # Type-Token Ratio (TTR)
        ttr = vocab_size / total_tokens if total_tokens > 0 else 0
        
        # Distinct n-grams
        distinct_metrics = {}
        for n in range(1, 4):  # 1-3 grams
            ngrams = []
            for pred in predictions:
                tokens = pred.lower().split()
                for i in range(len(tokens) - n + 1):
                    ngrams.append(' '.join(tokens[i:i+n]))
            
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            
            distinct_metrics[f'distinct_{n}'] = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        # Self-BLEU (measure of diversity - lower is more diverse)
        self_bleu = 0
        if NLTK_AVAILABLE and len(predictions) > 1:
            for i, pred in enumerate(predictions):
                other_preds = predictions[:i] + predictions[i+1:]
                pred_tokens = pred.lower().split()
                
                bleu_scores = []
                for other_pred in other_preds:
                    other_tokens = [other_pred.lower().split()]
                    try:
                        score = sentence_bleu(
                            other_tokens, pred_tokens,
                            smoothing_function=self.smoothing_function
                        )
                        bleu_scores.append(score)
                    except:
                        pass
                
                if bleu_scores:
                    self_bleu += sum(bleu_scores) / len(bleu_scores)
            
            self_bleu /= len(predictions)
        
        return {
            'vocab_size': vocab_size,
            'ttr': ttr,
            'self_bleu': self_bleu,
            **distinct_metrics
        }


def evaluate_model_outputs(
    predictions_file: str,
    references_file: str,
    vocabulary: Vocabulary
) -> Dict[str, Any]:
    """
    Evaluate model outputs from files.
    
    Args:
        predictions_file: Path to predictions file
        references_file: Path to references file
        vocabulary: Vocabulary object
        
    Returns:
        Dictionary of evaluation results
    """
    # Load predictions and references
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r') as f:
        references = [line.strip() for line in f]
    
    # Initialize metrics
    metrics = CaptionMetrics(vocabulary)
    
    # Compute metrics
    results = metrics.compute_metrics(predictions, references)
    
    # Add diversity metrics
    diversity_metrics = metrics.compute_diversity_metrics(predictions)
    results.update(diversity_metrics)
    
    return results
