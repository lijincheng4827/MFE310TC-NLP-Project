"""Main training pipeline, multi-seed evaluation, and figure generation."""
import argparse
import logging
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .data import LABELS, build_train_augmentations, load_splits, preprocess_baseline, preprocess_improved
from .edge_eval import build_edge_suite, evaluate_edge_cases, export_error_cases, metrics
from .models import FocalLoss, SentimentClassifier


class ReviewDataset(Dataset):
    """Tokenize reviews and create content and post-contrast masks."""
    def __init__(self, frame, tokenizer, max_length=256, improved=False):
        if max_length < 8:
            raise ValueError('max_length must be at least eight')
        self.labels = frame.label.to_list()
        self.items = []
        for text in frame.text:
            text = preprocess_improved(text) if improved else text
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded['input_ids'], encoded['offset_mapping']
            budget = max_length - 2
            # Keep both ends so a late change in opinion remains visible.
            keep = list(range(len(ids)))
            if len(ids) > budget:
                head = budget // 2
                keep = keep[:head] + keep[-(budget - head):] if improved else keep[:budget]
            markers = list(re.finditer(r'\b(?:however|but|yet|nevertheless)\b', text, re.I))
            end = markers[-1].end() if markers else len(text) + 1
            selected = [ids[i] for i in keep]
            selected_offsets = [offsets[i] for i in keep]
            values = tokenizer.build_inputs_with_special_tokens(selected)
            layout = tokenizer.build_inputs_with_special_tokens([-1] * len(selected))
            special = [int(token != -1) for token in layout]
            content, contrast = [], []
            j = 0
            for flag in special:
                content.append(int(not flag))
                contrast.append(int(not flag and selected_offsets[j][0] >= end) if not flag else 0)
                j += int(not flag)
            pad = max_length - len(values)
            self.items.append({
                'input_ids': torch.tensor(values + [tokenizer.pad_token_id] * pad),
                'attention_mask': torch.tensor([1] * len(values) + [0] * pad),
                'content_mask': torch.tensor(content + [0] * pad),
                'contrast_mask': torch.tensor(contrast + [0] * pad)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return {**self.items[index], 'labels': torch.tensor(self.labels[index])}


def seed_all(seed):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, tokenizer, texts, config, improved=True, attention=False):
    """Run batch inference and return class probabilities (and optional attention weights)."""
    frame = pd.DataFrame({'text': list(texts), 'label': 0})
    dataset = ReviewDataset(frame, tokenizer, config['max_length'], improved)
    device = next(model.parameters()).device
    probs, weights = [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=config['batch_size']):
            batch.pop('labels')
            logits, attn = model(**{k: v.to(device) for k, v in batch.items()}, return_attention=True)
            probs.append(logits.softmax(-1).cpu().numpy())
            weights.append(attn.cpu().numpy())
    values = np.concatenate(probs)
    return (values, np.concatenate(weights), dataset) if attention else values


def train_neural(splits, config, seed, improved, tokenizer):
    """Fine-tune DistilBERT model with early stopping based on validation Macro-F1."""
    seed_all(seed)
    model = SentimentClassifier(config['model_name'], improved)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    frame = splits['train'].copy()
    if improved:
        frame = pd.concat([frame, build_train_augmentations(seed)], ignore_index=True)
    dataset = ReviewDataset(frame, tokenizer, config['max_length'], improved)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    ce = nn.CrossEntropyLoss(label_smoothing=0.05 if improved else 0)
    focal = FocalLoss(1.5)
    history, best, best_state, stale = [], -1, None, 0
    for epoch in range(config['epochs']):
        model.train()
        total = 0
        for batch in loader:
            labels = batch.pop('labels').to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{k: v.to(device) for k, v in batch.items()})
            # Combine label-smoothed cross-entropy (80%) and focal loss (20%) for the improved model
            loss = 0.8 * ce(logits, labels) + 0.2 * focal(logits, labels) if improved else ce(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        val = predict(model, tokenizer, splits['validation'].text, config, improved)
        score = metrics(splits['validation'].label, val.argmax(1))
        history.append(dict(model='improved' if improved else 'principal', seed=seed,
                            epoch=epoch + 1, loss=total / len(dataset), **score))
        logging.info('%s', history[-1])
        if score['macro_f1'] > best:
            best, stale = score['macro_f1'], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    return model, history


def plot_all(splits, main, edge, matrices, attention_samples, out):
    """Render all 5 evaluation figures at 300 DPI into the results folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})

    def save(fig, filename, title):
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / filename, dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dist = pd.DataFrame({k: v.label.value_counts().reindex(range(3), fill_value=0)
                         for k, v in splits.items()}, index=range(3))
    dist.index = LABELS
    dist.plot.bar(ax=axes[0], rot=0)
    axes[0].set_ylabel('Reviews')
    for name, frame in splits.items():
        axes[1].hist(frame.text.str.split().str.len(), bins=25, alpha=0.5, label=name)
    axes[1].set_xlabel('Words per review')
    axes[1].legend()
    save(fig, 'fig1_data_dist.png', 'Dataset Label Distribution and Review Length')

    fig, ax = plt.subplots(figsize=(8, 4))
    main.set_index('model')[['accuracy', 'macro_f1', 'weighted_f1']].plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1.08)
    save(fig, 'fig2_model_comp.png', 'Test Set Performance Across Models')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    save(fig, 'fig3_conf_matrix.png', 'Confusion Matrices on Held-Out Test Set')

    fig, ax = plt.subplots(figsize=(12, 5))
    edge.pivot(index='category', columns='model', values='accuracy').plot.bar(ax=ax, rot=15)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    ax.set_ylim(0, 1.15)
    save(fig, 'fig4_edge_acc.png', 'Model Accuracy Across Six Restaurant Edge-Case Categories')

    fig, axes = plt.subplots(len(attention_samples), 1, figsize=(12, 2.8 * len(attention_samples)), squeeze=False)
    for ax, (words, values) in zip(axes.flat, attention_samples):
        count = min(24, len(words))
        indices = np.argsort(values)[-count:]
        indices.sort()
        ax.bar(range(count), np.asarray(values)[indices])
        ax.set_xticks(range(count), np.asarray(words)[indices], rotation=45, ha='right')
        ax.set_ylabel('Weight')
    save(fig, 'fig5_attn_viz.png', 'Word-Level Attention Weights on Sample Reviews')


def word_attention(tokenizer, item, weights):
    """Merge WordPiece subword attention weights back into full words."""
    tokens = tokenizer.convert_ids_to_tokens(item['input_ids'].tolist())
    words, values = [], []
    for token, weight, content in zip(tokens, weights, item['content_mask']):
        if not content:
            continue
        if token.startswith('##') and words:
            words[-1] += token[2:]
            values[-1] += float(weight)
        else:
            words.append(token)
            values.append(float(weight))
    return words, values


def train_and_eval(config, splits=None, tokenizer=None):
    """Run all three models across seeds, save CSV tables, and generate figures."""
    from .demo import DEMOS, run_demo
    out = Path('results')
    out.mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    splits = load_splits(config) if splits is None else splits
    for name, frame in splits.items():
        counts = frame.label.value_counts().reindex(range(3), fill_value=0)
        logging.info('%s: %s reviews; %s', name, len(frame),
                     {label: int(counts[i]) for i, label in enumerate(LABELS)})
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True) if tokenizer is None else tokenizer
    suite = build_edge_suite()
    main_rows, edge_rows, errors, history = [], [], [], []
    matrices, samples = {}, []
    if 42 not in config['seeds']:
        raise ValueError('seeds must include 42 for the saved demo')
    for seed in config['seeds']:
        baseline = make_pipeline(TfidfVectorizer(preprocessor=preprocess_baseline,
                                                ngram_range=(1, 2), max_features=30000),
                                 LogisticRegression(max_iter=1000, random_state=seed))
        baseline.fit(splits['train'].text, splits['train'].label)
        for name in ['baseline', 'principal', 'improved']:
            logging.info('Training %s, seed %s', name, seed)
            if name == 'baseline':
                pred = baseline.predict(splits['test'].text)
                edge_pred = baseline.predict(suite.text)
            else:
                model, rows = train_neural(splits, config, seed, name == 'improved', tokenizer)
                history.extend(rows)
                pred = predict(model, tokenizer, splits['test'].text, config, name == 'improved').argmax(1)
                edge_pred = predict(model, tokenizer, suite.text, config, name == 'improved').argmax(1)
            main_rows.append(dict(model=name, seed=seed, **metrics(splits['test'].label, pred)))
            edge_rows.append(evaluate_edge_cases(suite, edge_pred, name, seed))
            errors.append(export_error_cases(suite, edge_pred, name, seed))
            main_errors = export_error_cases(splits['test'].assign(category='main'), pred, name, seed)
            errors.append(main_errors)
            if seed == 42:
                matrices[name] = confusion_matrix(splits['test'].label, pred, labels=[0, 1, 2])
                if name == 'improved':
                    torch.save({'state': {k: v.cpu() for k, v in model.state_dict().items()},
                                'config': config}, 'checkpoints/improved.pt')
                    run_demo(model, tokenizer, config, out / 'demo_out.txt')
                    _, weights, dataset = predict(model, tokenizer, [DEMOS[0], DEMOS[5]], config, attention=True)
                    samples = [word_attention(tokenizer, item, weight) for item, weight in zip(dataset.items, weights)]
            if name != 'baseline':
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    main_seeds, edge_seeds = pd.DataFrame(main_rows), pd.concat(edge_rows, ignore_index=True)
    main = main_seeds.groupby('model', sort=False)[['accuracy', 'macro_f1', 'weighted_f1']].agg(['mean', 'std'])
    main.columns = ['_'.join(c) for c in main.columns]
    edge = edge_seeds.groupby(['model', 'category'], sort=False)[['accuracy', 'macro_f1', 'weighted_f1']].agg(['mean', 'std'])
    edge.columns = ['_'.join(c) for c in edge.columns]
    main.to_csv(out / 'main_metrics.csv')
    edge.to_csv(out / 'edge_metrics.csv')
    main_seeds.to_csv(out / 'main_metrics_seeds.csv', index=False)
    edge_seeds.to_csv(out / 'edge_metrics_seeds.csv', index=False)
    pd.concat(errors, ignore_index=True).to_csv(out / 'error_cases.csv', index=False)
    pd.DataFrame(history).to_csv(out / 'train_history.csv', index=False)
    plot_all(splits, main_seeds.groupby('model', sort=False).mean(numeric_only=True).reset_index(),
             edge_seeds.groupby(['model', 'category'], sort=False).mean(numeric_only=True).reset_index(),
             matrices, samples, out)


def main():
    """Parse arguments, initialize logging, and start the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler('logs/experiment.log', mode='w'), logging.StreamHandler()])
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logging.info('Device: %s', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_and_eval(config)


if __name__ == '__main__':
    main()
