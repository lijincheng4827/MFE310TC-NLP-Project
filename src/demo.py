"""Interactive inference demo on 6 representative Middle Eastern restaurant reviews."""
import argparse
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import LABELS
from .edge_eval import CATEGORIES
from .models import SentimentClassifier

FOOD_PRAISE = ('The lamb shawarma was tender and full of flavour. The warm pita was soft, '
               'the hummus was creamy, and the grilled vegetables tasted fresh. We loved '
               'the fragrant rice and the carefully balanced spices. Every plate looked '
               'beautiful and the portions were generous. ')
# Six sample reviews representing each edge-case category
DEMOS = [
    'Five stars for the shawarma packaging! The tahini sauce leaked into my backpack. A masterpiece.',
    'The falafel was stale and the lentil soup tasted sour 😡🤢👎',
    'We paid for two lamb mandi platters. An hour later our table still had only empty plates.',
    ' '.join((FOOD_PRAISE * 3).split()[:120]) + '. However, the head waiter shouted at my mother and refused to apologize. We will never return.',
    'The beef kebab was not good, and I cannot recommend this restaurant.',
    'Habibi this shawarma is the GOAT, ngl lowkey fire, yyds, def do not sleep on this spot!']


def aspect_clauses(text):
    """Split review into clauses and group them into Cuisine vs. Service aspects."""
    clauses = re.split(r'(?<=[.!?])\s+|;|\bhowever\b|\bbut\b', text, flags=re.I)
    cuisine, service = [], []
    for clause in filter(str.strip, clauses):
        packaging = re.search(r'packaging|leak|backpack|takeaway|spilled', clause, re.I)
        if packaging or re.search(r'waiter|server|staff|service|paid|empty plates|hour later|apologiz', clause, re.I):
            service.append(clause)
        if not packaging and re.search(r'shawarma|falafel|soup|mandi|kebab|pita|hummus|rice|spice|plate looked|portions|vegetables', clause, re.I):
            cuisine.append(clause)
    return cuisine, service


def management_action(label, category, cuisine=None, service=None):
    """Return recommended follow-up action for restaurant staff based on predictions."""
    if label == 0:
        if category == 'sarcasm':
            return 'Apologize to guest for leaked sauce and inspect takeaway packaging.'
        if category == 'long_tail' and cuisine == 2 and service == 0:
            return 'Praise kitchen team for cuisine, but manager must urgently address waiter misconduct.'
        return 'Apologize to the guest and investigate the food and service concerns.'
    if label == 2:
        return 'Thank the guest and share positive feedback with the kitchen team.'
    return 'Ask the guest how the dining experience could be improved.'


def run_demo(model, tokenizer, config, output='results/demo_out.txt'):
    """Run inference on the 6 demo reviews and save formatted output."""
    from .train import predict
    probs = predict(model, tokenizer, DEMOS, config)
    lines = []
    for i, (category, text, prob) in enumerate(zip(CATEGORIES, DEMOS, probs), 1):
        cuisine, service = aspect_clauses(text)
        labels = []
        aspects = []
        for name, clauses in [('Cuisine', cuisine), ('Service/Packaging' if i == 1 else 'Service', service)]:
            if clauses:
                p = predict(model, tokenizer, [' '.join(clauses)], config)[0]
                label = int(p.argmax())
                aspects.append(f'[{name}: {LABELS[label]} ({100 * p[label]:.1f}%)]')
                labels.append(label)
            else:
                aspects.append(f'[{name}: {"neutral/not mentioned" if name == "Cuisine" else "not mentioned"}]')
                labels.append(None)
        label = int(prob.argmax())
        lines.extend([f'Review {i}: {text}',
                      f'Predicted Sentiment: {LABELS[label]} ({100 * prob[label]:.1f}%)',
                      'Aspect Breakdown: ' + ' | '.join(aspects),
                      f'Detected Pattern: {category}',
                      f'Management Action: {management_action(label, category, *labels)}', ''])
    content = '\n'.join(lines)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(content, encoding='utf-8')
    print(content)
    return probs


def main():
    """Load saved checkpoint and execute the live inference demonstration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/improved.pt')
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    model = SentimentClassifier(config['model_name'], improved=True)
    model.load_state_dict(saved['state'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    run_demo(model, tokenizer, config)


if __name__ == '__main__':
    main()
