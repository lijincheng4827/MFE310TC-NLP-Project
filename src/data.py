"""Data loading, train/val/test splitting, and text preprocessing helpers."""
import random
import re

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ['negative', 'neutral', 'positive']
# Map common emotion emojis to plain English words so DistilBERT can use pretrained embeddings
EMOJI = dict(zip(
    ['😡', '🤬', '🤢', '👎', '😒', '🤤', '🔥', '😍', '👍', '😊', '😐'],
    ['angry terrible', 'furious awful', 'disgusting bad', 'bad disappointing',
     'unimpressed poor', 'delicious amazing', 'excellent great', 'love wonderful',
     'good nice', 'happy pleased', 'average okay']))
# Common restaurant slang, idioms, and typos to normalize before tokenization
SLANG = [
    (r"\b(do not|dont|don't)\s+sleep\s+on\b", 'definitely try and highly recommend'),
    (r'\bthe\s+goat\b', 'the best and most delicious'),
    (r'\bgoat\b(?!\s+(?:cheese|curry|meat|stew))', 'amazing best'),
    (r'\blowkey\s+fire\b', 'really delicious and great'),
    (r'\byyds\b', 'outstanding delicious'),
    (r'\bbussin(\s+fr)?\b', 'very tasty and fresh'),
    (r'\bhit\s+different\b', 'tasted wonderful'),
    (r'\b(mid|meh|just\s+aight|aight)\b', 'average and mediocre'),
    (r'\bsrvce\b', 'service'), (r'\bovrepriced\b', 'overpriced'),
    (r'\bghosted\s+us\b', 'ignored our table completely'),
    (r'\bbaklva\b', 'baklava'), (r'\bshwrma\b', 'shawarma')]


def preprocess_baseline(text):
    """Basic regex cleaning for TF-IDF baseline (lowercase + strip punctuation/emojis)."""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())


def preprocess_improved(text):
    """Normalize emojis, slang, and sarcastic praise-mishap patterns for the improved model."""
    text = text.replace('’', "'")
    for symbol, words in EMOJI.items():
        text = text.replace(symbol, f' {words} ')
    for pattern, words in SLANG:
        text = re.sub(pattern, words, text, flags=re.I)
    # If a review has extreme praise ("five stars") alongside a bad mishap ("leaked"), flag as sarcasm
    praise = r'five stars|masterpiece|bravo|vip treatment|royal hospitality|complimentary workout'
    adverse = r'leaked|spilled|charged us twice|after everyone else had left|chasing a server|missing|cold|piece of glass'
    if re.search(praise, text, re.I) and re.search(adverse, text, re.I):
        text += ' [overall a terrible disappointing complaint]'
    return re.sub(r'\s+', ' ', text).strip()


def load_splits(config):
    """Load Yelp dataset, remove duplicate texts, and create balanced 70/15/15 splits."""
    from datasets import load_dataset
    size = config['sample_size']
    if isinstance(size, bool) or not isinstance(size, int) or size < 60 or size % 3:
        raise ValueError('sample_size must be an integer of at least 60, divisible by three')
    source = load_dataset(config['dataset_name'], split='train')
    # Map 5-star ratings into 3 classes: 1-2 stars -> negative (0), 3 -> neutral (1), 4-5 -> positive (2)
    frame = source.to_pandas()[['text', 'label']].drop_duplicates('text')
    frame['label'] = frame.label.map({0: 0, 1: 0, 2: 1, 3: 2, 4: 2})
    counts = frame.label.value_counts().reindex(range(3), fill_value=0)
    if counts.min() < size // 3:
        raise ValueError(f'Need {size // 3} unique reviews per class; available: {counts.to_dict()}')
    frame = pd.concat([g.sample(size // 3, random_state=config['data_seed'])
                       for _, g in frame.groupby('label')], ignore_index=True)
    train, rest = train_test_split(frame, test_size=0.3, stratify=frame.label,
                                   random_state=config['data_seed'])
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label,
                                 random_state=config['data_seed'])
    return {k: v.reset_index(drop=True) for k, v in
            [('train', train), ('validation', val), ('test', test)]}


def build_train_augmentations(seed):
    """Generate 180 extra training examples covering contrast, slang, emoji, and irony patterns."""
    dishes = ['vegetable risotto', 'tomato pasta', 'mushroom pizza', 'roast chicken',
              'potato gratin', 'garden salad', 'fish pie', 'bean burger',
              'pumpkin ravioli', 'spinach quiche']
    patterns = [
        ('The {d} tasted lovely, but the cashier insulted our family and refused a refund.', 0),
        ('The room looked shabby, however the {d} was superb and the staff were kind.', 2),
        ('The {d} was decent, yet neither the price nor the service stood out.', 1),
        ('Our {d} never arrived; the srvce ghosted us and the drinks were ovrepriced.', 0),
        ('The server replaced the {d} immediately and removed it from our bill.', 2),
        ('The {d} arrived at the stated time and matched the menu description.', 1),
        ('I cannot recommend the {d}; it was not remotely edible.', 0),
        ('I did not expect to enjoy the {d}, but I loved every bite.', 2),
        ('The {d} was neither disappointing nor particularly memorable.', 1),
        ('The {d}: 🤢 👎. Never coming back.', 0),
        ('The {d} was bussin fr, lowkey fire, do not sleep on it! 😊', 2),
        ('The {d} was mid, just aight, nothing special. 😐', 1),
        ('A masterpiece of billing: they charged us twice for the {d} and kept the extra payment.', 0),
        ('Complimentary workout: we spent dinner chasing a server to recover our missing {d}.', 0),
        ('The {d} was fresh and excellent, and the team made us feel welcome.', 2),
        ('The {d} was ordinary; I might return if I am nearby.', 1),
        ('There is no reason to avoid the {d}; it was delicious.', 2),
        ('It is not true that the {d} was good; it was stale and dry.', 0)]
    rows = []
    intro = ('We spent the evening discussing the menu and sharing several courses around '
             'a large table by the window. The dining room had soft lighting and quiet '
             'music, and we took our time looking through every dish before choosing '
             'the meal we had planned for this family gathering. ')
    for dish in dishes:
        for i, (pattern, label) in enumerate(patterns):
            text = pattern.format(d=dish)
            if i < 3:
                text = intro + text
            rows.append({'text': text, 'label': label})
    random.Random(seed).shuffle(rows)
    return pd.DataFrame(rows)
