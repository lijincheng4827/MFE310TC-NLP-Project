"""Evaluation suite for the 6 Middle Eastern restaurant edge-case categories."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CATEGORIES = ['sarcasm', 'emoji', 'implicit', 'long_tail', 'negation', 'slang_oov']
# Separate solid dishes and liquid items so spill/leak templates sound natural
DISHES = ['lamb shawarma', 'pistachio kunafa', 'beef kebab', 'falafel platter',
          'hummus with warm pita', 'lamb mandi', 'baklava', 'grilled halloumi']
LIQUIDS = ['hot lentil soup', 'mint tea', 'tahini sauce', 'garlic toum',
           'pomegranate stew', 'lentil soup', 'rose lemonade', 'yogurt sauce']


def build_edge_suite():
    """Build 240 test reviews (40 per category across 8 dishes x 5 templates)."""
    patterns = {
        'sarcasm': [
            ('Five stars for the takeaway box: the {s} leaked all over my coat. A masterpiece.', 0),
            ('Bravo, our {d} arrived cold after everyone else had left. Royal hospitality.', 0),
            ('VIP treatment means chasing a server while our {d} goes missing, apparently.', 0),
            ('I joked that the {d} would be a disaster. Instead it was delicious and the service excellent.', 2),
            ('The menu called the {d} a life-changing experience. It was just an ordinary lunch.', 1)],
        'emoji': [
            ('The {d} was stale and the {s} tasted sour 😡🤢👎', 0),
            ('This {d} 😍🤤🔥 and such friendly staff 😊👍', 2),
            ('The {d} 😐. Fine for a quick meal, nothing special.', 1),
            ('Our {s} leaked in the bag, staff shrugged 😒🤬', 0),
            ('Fresh {d}, generous portions, lovely service 😍👍', 2)],
        'implicit': [
            ('We paid for the {d}. Forty minutes later our table still had only cutlery.', 0),
            ('The receipt lists two {d} portions. Only one reached our table.', 0),
            ('The server remembered our allergy and brought a separate plate of {d} without being asked.', 2),
            ('The {d} arrived in twelve minutes, as stated on the menu.', 1),
            ('We finished the {d} and immediately reserved the same table for next Friday.', 2)],
        'long_tail': [
            ('{p} However, the waiter shouted at my mother and refused to apologize. We will never return.', 0),
            ('{p} But the {s} spilled on my child and the manager blamed us. The visit was awful.', 0),
            ('{n} Nevertheless, the manager listened, replaced the meal and made the evening wonderful.', 2),
            ('{n} Yet the {d} was outstanding and the team went out of their way to help. We loved it.', 2),
            ('{p} However, the price and long wait offset the good food. Overall it was average.', 1)],
        'negation': [
            ('The {d} was not good, and I cannot recommend this restaurant.', 0),
            ('I would not say the {d} was bad; it was delicious and I will return.', 2),
            ('The {d} was neither excellent nor awful, just ordinary.', 1),
            ('Do not believe the praise: the {d} was inedible and the service was not acceptable.', 0),
            ('Not once did the waiter ignore us. The {d} was excellent.', 2)],
        'slang_oov': [
            ('Habibi, the {d} is the GOAT, lowkey fire, yyds. Dont sleep on this place!', 2),
            ('The {d} was bussin fr and the {s} hit different. Loved it.', 2),
            ('The {d} was mid, just aight, meh. An average meal.', 1),
            ('Srvce ghosted us, {d} was ovrepriced and dry. Never again.', 0),
            ('The {d} was burnt, baklva stale, srvce rude. Worst meal this week.', 0)]}
    rows = []
    for category, templates in patterns.items():
        for i, dish in enumerate(DISHES):
            praise = (f'The {dish} was delicious and beautifully prepared. '
                      'The bread was warm, the herbs were fresh, and every bite was full of flavour. '
                      'We enjoyed the generous portions and carefully balanced spices. ')
            poor = (f'The {dish} arrived late and the bread seemed dry. '
                    'The room was noisy and our table was too close to the entrance. '
                    'At first we regretted choosing this restaurant for our family dinner. ')
            for template, label in templates:
                text = template.format(d=dish, s=LIQUIDS[i],
                                       p=(praise * 4).strip(), n=(poor * 4).strip())
                rows.append({'category': category, 'text': text, 'label': label})
    return pd.DataFrame(rows)


def metrics(gold, pred):
    """Compute accuracy, active-class macro-F1, and weighted-F1."""
    return {'accuracy': accuracy_score(gold, pred),
            'macro_f1': f1_score(gold, pred, labels=np.unique(gold), average='macro', zero_division=0),
            'weighted_f1': f1_score(gold, pred, average='weighted', zero_division=0)}


def evaluate_edge_cases(suite, pred, model, seed):
    """Evaluate model predictions across each of the 6 edge-case categories."""
    frame = suite.assign(pred=np.asarray(pred))
    return pd.DataFrame([dict(model=model, seed=seed, category=name,
                              **metrics(group.label, group.pred))
                         for name, group in frame.groupby('category', sort=False)])


def export_error_cases(suite, pred, model, seed):
    """Filter misclassified examples for error analysis."""
    frame = suite.assign(pred=np.asarray(pred), model=model, seed=seed)
    return frame.loc[frame.label != frame.pred].copy()
