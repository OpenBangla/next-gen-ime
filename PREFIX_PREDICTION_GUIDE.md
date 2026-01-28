# Prefix-Based Prediction for Bangla Next Word Model

## Overview

Two methods to filter predictions by prefix (e.g., only show words starting with 'কা'):

### Method 1: Post-Filtering (Simple)
```python
predictions = predictor.predict_with_prefix(
    text="আমি বাংলায়",
    prefix="কা",
    top_k=10,
    search_top_n=100  # Search through top 100 candidates
)
```

**How it works:**
1. Get top N predictions (e.g., 100)
2. Filter those that start with the prefix
3. Return top K matches

**Pros:**
- Simple and straightforward
- Easy to understand and debug
- Flexible - can apply complex filtering logic

**Cons:**
- Might not find enough matches if prefix is rare
- Processes tokens unnecessarily


### Method 2: Logit Masking (Advanced)
```python
predictions = predictor.predict_with_prefix_advanced(
    text="আমি বাংলায়",
    prefix="কা",
    top_k=10
)
```

**How it works:**
1. Get model logits for all vocabulary
2. Mask (set to -inf) tokens that don't match prefix
3. Apply softmax to get probabilities
4. Return top K

**Pros:**
- More efficient for large vocabularies
- Guaranteed to find all matches
- Re-normalized probabilities only for matching tokens

**Cons:**
- Slower for small prefixes (must check all vocab tokens)
- More complex implementation


## Usage Examples

### Example 1: Basic Usage
```python
from prefix_predictor import PrefixAwarePredictor

# Load your model (see use_prefix_prediction.py for details)
predictor = PrefixAwarePredictor(model, tokenizer, device, context_len)

# Get predictions starting with 'কা'
predictions = predictor.predict_with_prefix("আমি বাংলায়", prefix="কা", top_k=5)

for word, prob in predictions:
    print(f"{word}: {prob*100:.1f}%")
```

Output:
```
কাজ: 15.3%
কাল: 12.8%
কারণ: 9.2%
কাছে: 7.5%
কাছ: 5.1%
```

### Example 2: Interactive Mode
```python
predictor.interactive_with_prefix()
```

Then type:
```
📝 Enter text [| prefix]: আমি বাংলায় | কা
```

Output:
```
   Filtering with prefix: 'কা'

🎯 Predictions:
    1. কাজ (15.3%)
    2. কাল (12.8%)
    3. কারণ (9.2%)
    4. কাছে (7.5%)
    5. কাছ (5.1%)
```

### Example 3: No Prefix (All Predictions)
```python
# Empty prefix returns all predictions
predictions = predictor.predict_with_prefix("আমি বাংলায়", prefix="", top_k=5)
```

### Example 4: Multiple Character Prefix
```python
# Can use multi-character prefixes
predictions = predictor.predict_with_prefix("বাংলাদেশের রাজধানী", prefix="ঢাক", top_k=5)
```

## When to Use Which Method?

### Use Method 1 (Post-Filtering) when:
- Vocabulary size is moderate (< 50K tokens)
- Prefix is relatively common
- You want simple, readable code
- You're prototyping

### Use Method 2 (Logit Masking) when:
- Vocabulary is very large (> 100K tokens)
- You need guaranteed prefix matches
- Performance is critical
- You want properly re-normalized probabilities


## Integration with Original Model

### Option A: Add to existing NextWordPredictor
Open `train_bangla_gru_sp.py` and add these methods to the `NextWordPredictor` class:

```python
def predict_with_prefix(self, text, prefix="", top_k=10, search_top_n=100):
    # Copy implementation from prefix_predictor.py
    ...
```

### Option B: Use as separate module
Keep `prefix_predictor.py` separate and import:

```python
from prefix_predictor import PrefixAwarePredictor

# Wrap your existing model
predictor = PrefixAwarePredictor(model, tokenizer, device, context_len)
```


## Common Patterns

### Pattern 1: Autocomplete with Prefix
```python
def autocomplete(context: str, user_typed: str, top_k: int = 5):
    """
    User has typed partial word after context
    
    Args:
        context: "আমি বাংলায়"
        user_typed: "কা"  (user is typing)
        top_k: Number of suggestions
    """
    return predictor.predict_with_prefix(context, prefix=user_typed, top_k=top_k)
```

### Pattern 2: Dynamic Filtering
```python
def get_suggestions_as_user_types(context: str, partial_word: str):
    """Update suggestions as user types each character"""
    suggestions = []
    
    for i in range(1, len(partial_word) + 1):
        prefix = partial_word[:i]
        preds = predictor.predict_with_prefix(context, prefix=prefix, top_k=3)
        suggestions.append((prefix, preds))
    
    return suggestions

# Example:
# context = "আমি বাংলায়"
# partial = "কাজ"
# 
# Results:
# "ক"  -> [কাজ, কাল, কারণ]
# "কা" -> [কাজ, কাল, কারণ]  (narrowed down)
# "কাজ" -> [কাজ]              (exact match)
```

### Pattern 3: Fallback Strategy
```python
def smart_predict(text: str, prefix: str = "", top_k: int = 10):
    """Try with prefix first, fallback to no prefix if no matches"""
    if prefix:
        results = predictor.predict_with_prefix(text, prefix=prefix, top_k=top_k)
        if results:
            return results
        else:
            print(f"No matches for prefix '{prefix}', showing all predictions")
    
    return predictor.predict_with_prefix(text, prefix="", top_k=top_k)
```


## Performance Tips

1. **Search window size**: Increase `search_top_n` if not finding enough matches:
   ```python
   predictor.predict_with_prefix(text, prefix="কা", search_top_n=200)
   ```

2. **Batch processing**: If checking multiple prefixes, reuse logits:
   ```python
   # Get logits once
   logits = model(x)
   
   # Try multiple prefixes
   for prefix in ["ক", "কা", "কাজ"]:
       # Apply different masks to same logits
       ...
   ```

3. **Caching**: For repeated queries, cache vocabulary prefix mapping:
   ```python
   # Build prefix index once
   prefix_index = {}
   for idx in range(vocab_size):
       piece = tokenizer.id_to_piece(idx)
       for i in range(1, len(piece) + 1):
           prefix = piece[:i]
           prefix_index.setdefault(prefix, []).append(idx)
   
   # Quick lookup
   matching_indices = prefix_index.get("কা", [])
   ```


## Troubleshooting

### Problem: No predictions found
- Check that prefix uses same script (Bangla characters: U+0980–U+09FF)
- Verify tokenizer is processing text correctly
- Increase `search_top_n` parameter
- Try shorter prefix (single character)

### Problem: Low probabilities
- This is normal with prefix filtering (re-normalization)
- Use `predict_with_prefix_advanced()` for proper re-normalization
- Consider temperature parameter to adjust distribution

### Problem: Slow performance
- Use Method 2 for large vocabularies
- Reduce `search_top_n` if using Method 1
- Ensure model is on GPU/MPS
- Use batch prediction if checking multiple texts


## Real-World Application Example

```python
class BanglaAutocomplete:
    def __init__(self, model_dir):
        self.model, self.tokenizer, self.device, self.context_len = load_model(model_dir)
        self.predictor = PrefixAwarePredictor(
            self.model, self.tokenizer, self.device, self.context_len
        )
    
    def suggest(self, full_text: str, cursor_position: int, num_suggestions: int = 5):
        """
        Suggest completions for word at cursor position
        
        Args:
            full_text: "আমি বাংলায় কা"
            cursor_position: 19 (after 'কা')
            num_suggestions: 5
        """
        # Split into context and partial word
        before_cursor = full_text[:cursor_position]
        
        # Find last word boundary
        words = before_cursor.split()
        if words:
            partial_word = words[-1]
            context = ' '.join(words[:-1])
        else:
            return []
        
        # Get predictions
        return self.predictor.predict_with_prefix(
            context, 
            prefix=partial_word, 
            top_k=num_suggestions
        )

# Usage in an app:
autocomplete = BanglaAutocomplete("bangla_gru_sp")
suggestions = autocomplete.suggest("আমি বাংলায় কা", cursor_position=19)
```
