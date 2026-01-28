"""
Modified NextWordPredictor with prefix filtering support
Add this to your train_bangla_gru_sp.py or use as a separate module
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class PrefixAwarePredictor:
    """Next word predictor with prefix filtering support"""
    
    def __init__(self, model, tokenizer, device, context_len):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_len = context_len
        self.model.eval()
    
    def predict_with_prefix(
        self, 
        text: str, 
        prefix: str = "", 
        top_k: int = 10,
        temperature: float = 1.0,
        search_top_n: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Predict next tokens with optional prefix filtering
        
        Args:
            text: Input text context
            prefix: Required prefix for predictions (e.g., 'কা')
            top_k: Number of results to return
            temperature: Sampling temperature
            search_top_n: How many top predictions to search through for prefix matches
        
        Returns:
            List of (token, probability) tuples matching the prefix
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) == 0:
            return []
        
        # Prepare context
        if len(tokens) > self.context_len:
            tokens = tokens[-self.context_len:]
        elif len(tokens) < self.context_len:
            tokens = [0] * (self.context_len - len(tokens)) + tokens
        
        x = torch.tensor([tokens], device=self.device)
        
        # Get more candidates to search through
        search_k = max(search_top_n, top_k * 5)
        probs, indices = self.model.predict_topk(x, search_k, temperature)
        
        results = []
        for idx, prob in zip(indices[0].tolist(), probs[0].tolist()):
            # Skip special tokens (pad=0, unk=1, bos=2, eos=3)
            if idx >= 4:
                piece = self.tokenizer.id_to_piece(idx)
                # Clean up SentencePiece's underscore prefix
                piece_clean = piece.replace('▁', '')
                
                if piece_clean:
                    # Check prefix match
                    if not prefix or piece_clean.startswith(prefix):
                        results.append((piece_clean, prob))
                        
                        # Stop when we have enough matches
                        if len(results) >= top_k:
                            break
        
        return results
    
    def predict_with_prefix_advanced(
        self, 
        text: str, 
        prefix: str = "", 
        top_k: int = 10,
        temperature: float = 1.0
    ) -> List[Tuple[str, float]]:
        """
        Advanced version: Mask non-matching tokens before selection
        This is more efficient as it filters at the logit level
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) == 0:
            return []
        
        # Prepare context
        if len(tokens) > self.context_len:
            tokens = tokens[-self.context_len:]
        elif len(tokens) < self.context_len:
            tokens = [0] * (self.context_len - len(tokens)) + tokens
        
        x = torch.tensor([tokens], device=self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.model(x) / max(temperature, 1e-8)
            
            # If prefix specified, mask non-matching tokens
            if prefix:
                # Create mask for tokens that match prefix
                mask = torch.zeros_like(logits[0], dtype=torch.bool)
                
                for idx in range(logits.size(-1)):
                    if idx >= 4:  # Skip special tokens
                        piece = self.tokenizer.id_to_piece(idx)
                        piece_clean = piece.replace('▁', '')
                        if piece_clean.startswith(prefix):
                            mask[idx] = True
                
                # Set non-matching tokens to very low probability
                logits[0, ~mask] = float('-inf')
            
            # Get probabilities and top-k
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
        
        results = []
        for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist()):
            if idx >= 4 and prob > 0:  # Skip special tokens and -inf probs
                piece = self.tokenizer.id_to_piece(idx)
                piece_clean = piece.replace('▁', '')
                if piece_clean:
                    results.append((piece_clean, prob))
        
        return results
    
    def demo_with_prefix(self):
        """Demo showing prefix filtering"""
        print("\n" + "=" * 60)
        print("  Prefix-Based Prediction Demo")
        print("=" * 60)
        
        examples = [
            ("আমি বাংলায়", ""),           # No prefix
            ("আমি বাংলায়", "গ"),          # Starts with 'গ'
            ("বাংলাদেশের রাজধানী", "ঢা"),  # Starts with 'ঢা'
            ("আজকে আমি", "কা"),           # Starts with 'কা'
            ("সে স্কুলে", "য"),            # Starts with 'য'
        ]
        
        for text, prefix in examples:
            print(f"\n📝 Input: {text}")
            if prefix:
                print(f"   Prefix filter: '{prefix}'")
            
            preds = self.predict_with_prefix(text, prefix=prefix, top_k=5)
            
            if preds:
                print("   Predictions:")
                for i, (word, prob) in enumerate(preds, 1):
                    bar = '█' * int(prob * 25)
                    print(f"   {i}. {word:12} {prob*100:5.1f}% {bar}")
            else:
                print("   No predictions found")
    
    def interactive_with_prefix(self):
        """Interactive mode with prefix support"""
        print("\n" + "=" * 60)
        print("  Interactive Mode with Prefix Filtering")
        print("  Format: <text> | <prefix>")
        print("  Example: আমি বাংলায় | গ")
        print("  Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n📝 Enter text [| prefix]: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Parse input
                if '|' in user_input:
                    text, prefix = user_input.split('|', 1)
                    text = text.strip()
                    prefix = prefix.strip()
                else:
                    text = user_input
                    prefix = ""
                
                if prefix:
                    print(f"   Filtering with prefix: '{prefix}'")
                
                preds = self.predict_with_prefix(text, prefix=prefix, top_k=10)
                
                if preds:
                    print("\n🎯 Predictions:")
                    for i, (word, prob) in enumerate(preds, 1):
                        print(f"   {i:2d}. {word} ({prob*100:.1f}%)")
                else:
                    print(f"   No predictions found" + (f" starting with '{prefix}'" if prefix else ""))
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break


# Usage example
if __name__ == "__main__":
    print("To use this with your trained model:")
    print()
    print("from prefix_predictor import PrefixAwarePredictor")
    print()
    print("# Load your model and tokenizer")
    print("predictor = PrefixAwarePredictor(model, tokenizer, device, context_len)")
    print()
    print("# Get predictions with prefix")
    print("predictions = predictor.predict_with_prefix('আমি বাংলায়', prefix='কা', top_k=5)")
    print()
    print("# Or use interactive mode")
    print("predictor.interactive_with_prefix()")
