"""Unit and integration tests verifying preprocessing, model pooling, and pipeline outputs."""
import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertModel

from src.data import build_train_augmentations, preprocess_improved, preprocess_baseline
from src.demo import DEMOS, aspect_clauses, management_action
from src.edge_eval import CATEGORIES, build_edge_suite, metrics
from src.models import FocalLoss, SentimentClassifier
from src.train import ReviewDataset, predict, train_and_eval


@pytest.fixture
def tokenizer(tmp_path):
    """Build a lightweight local WordPiece tokenizer from project texts for fast CPU testing."""
    texts = list(build_train_augmentations(42).text) + list(build_edge_suite().text) + DEMOS
    words = sorted(set(re.findall(r"[a-z]+|[^\w\s]", ' '.join(preprocess_improved(t).lower() for t in texts))))
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + words))
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


def tiny_encoder(size):
    """Create a 1-layer tiny DistilBertModel so unit tests run quickly without downloading weights."""
    return DistilBertModel(DistilBertConfig(vocab_size=size, dim=24, hidden_dim=48,
                                         n_layers=1, n_heads=2, dropout=0, attention_dropout=0))


@pytest.mark.parametrize('text,expected', [
    (DEMOS[0], 'overall a terrible disappointing complaint'),
    (DEMOS[1], 'disgusting bad'), (DEMOS[2], 'empty plates'),
    (DEMOS[3], 'However'), (DEMOS[4], 'cannot recommend'),
    (DEMOS[5], 'definitely try and highly recommend'),
    ('the GOAT, lowkey fire, yyds', 'the best and most delicious'),
    ('srvce ghosted us, ovrepriced shwrma and baklva', 'ignored our table completely'),
    ('goat cheese with pita', 'goat cheese'),
    ('mint tea 😊😐', 'happy pleased average okay')])
def test_normalization(text, expected):
    """Verify emoji, slang, and sarcasm normalization rules in preprocess_improved."""
    assert expected in preprocess_improved(text)
    assert '😡' not in preprocess_baseline(text)


def test_suite_and_augmentation():
    """Check that the 240-review edge suite and 180 augmentations are balanced and disjoint."""
    suite = build_edge_suite()
    assert len(suite) == 240
    assert suite.text.nunique() == 240
    assert suite.groupby('category').size().to_dict() == dict.fromkeys(CATEGORIES, 40)
    assert suite.groupby('category').label.nunique().eq(3).all()
    augment = build_train_augmentations(42)
    assert len(augment) == 180
    assert len(set(augment.text)) == 180
    assert not set(augment.text) & set(suite.text)
    pd.testing.assert_frame_equal(augment, build_train_augmentations(42))
    assert not set(DEMOS) & set(augment.text)
    assert augment.text.str.split().str.len().gt(40).sum() == 30
    assert sum('overall a terrible disappointing complaint' in preprocess_improved(t) for t in augment.text) == 20
    for text in suite.text:
        assert not re.search(r'spilled hot (baklava|shawarma)', text)


def test_active_f1():
    """Verify that macro_f1 only averages over classes present in gold labels."""
    values = metrics([0] * 20, [0] * 19 + [2])
    assert values['accuracy'] == 0.95
    assert values['macro_f1'] > 0.97
    assert values['weighted_f1'] > 0.97


def test_final_contrast_mask(tokenizer):
    """Ensure contrast_mask highlights tokens after the final contrast marker."""
    text = 'Good soup but poor service however delicious bread yet rude staff nevertheless wonderful food'
    data = ReviewDataset(pd.DataFrame({'text': [text], 'label': [2]}), tokenizer, 64, True)
    item = data[0]
    assert tokenizer.decode(item['input_ids'][item['contrast_mask'].bool()]) == 'wonderful food'
    assert not (item['contrast_mask'].bool() & ~item['content_mask'].bool()).any()


def test_long_review_keeps_tail(tokenizer):
    """Confirm head-tail truncation preserves the ending complaint in long reviews."""
    text = 'delicious bread ' * 180 + 'however the waiter was rude'
    item = ReviewDataset(pd.DataFrame({'text': [text], 'label': [0]}), tokenizer, 64, True)[0]
    assert 'waiter was rude' in tokenizer.decode(item['input_ids'])
    assert item['contrast_mask'].sum() > 0
    plain = ReviewDataset(pd.DataFrame({'text': [text], 'label': [0]}), tokenizer, 64, False)[0]
    assert 'waiter' not in tokenizer.decode(plain['input_ids'])


def test_pooling_equation():
    """Verify the 0.25 [CLS] + 0.75 contrast pooling fusion math on long contrast reviews."""
    class Encoder(nn.Module):
        config = SimpleNamespace(hidden_size=3)

        def forward(self, input_ids, attention_mask):
            states = torch.arange(input_ids.numel() * 3, dtype=torch.float).reshape(*input_ids.shape, 3) / 100
            return SimpleNamespace(last_hidden_state=states)

    model = SentimentClassifier('', True, Encoder()).eval()
    model.classifier = nn.Identity()
    nn.init.zeros_(model.res_proj.weight)
    nn.init.zeros_(model.res_proj.bias)
    ids = torch.zeros((2, 50), dtype=torch.long)
    content = torch.ones_like(ids)
    contrast = torch.zeros_like(ids)
    contrast[0, 45:] = 1
    states = model.encoder(ids, content).last_hidden_state
    out, attn = model(ids, content, content, contrast, True)
    torch.testing.assert_close(out[0], 0.25 * states[0, 0] + 0.75 * states[0, 45:].mean(0))
    torch.testing.assert_close(out[1], states[1, 0])
    torch.testing.assert_close(attn.sum(1), torch.ones(2))


def test_empty_and_gradient(tokenizer):
    """Check that empty strings do not produce NaN losses and gradients flow through new layers."""
    model = SentimentClassifier('', True, tiny_encoder(len(tokenizer)))
    data = ReviewDataset(pd.DataFrame({'text': ['', 'good soup but rude staff'], 'label': [1, 0]}), tokenizer, 32, True)
    batch = {k: torch.stack([data[i][k] for i in range(2)]) for k in data[0]}
    labels = batch.pop('labels')
    logits = model(**batch)
    loss = 0.8 * nn.CrossEntropyLoss(label_smoothing=0.05)(logits, labels) + 0.2 * FocalLoss()(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.attn_proj.weight.grad is not None
    assert torch.isfinite(model.res_proj.weight.grad).all()


def test_aspects_and_actions():
    """Verify clause splitting and management action recommendations on demo examples."""
    assert not aspect_clauses(DEMOS[0])[0]
    assert 'leaked' in ' '.join(aspect_clauses(DEMOS[0])[1])
    food, service = aspect_clauses(DEMOS[3])
    assert 'shouted' not in ' '.join(food)
    assert 'shouted' in ' '.join(service)
    assert not aspect_clauses(DEMOS[5])[1]
    assert 'waiter misconduct' in management_action(0, 'long_tail', 2, 0)
    assert 'leaked sauce' in management_action(0, 'sarcasm')
    assert 'kitchen team' in management_action(2, 'slang_oov')


def test_end_to_end(tmp_path, monkeypatch, tokenizer):
    """Run a full end-to-end smoke test checking CSVs, 300 DPI figures, checkpoint, and zip archive."""
    from PIL import Image
    from src import train
    root = Path(__file__).resolve().parents[1]
    for name in ['README.md', 'config.yaml', 'requirements.txt', 'run_all.sh',
                 'tests/test_pipeline.py'] + [str(p.relative_to(root)) for p in (root / 'src').glob('*.py')]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, 'SentimentClassifier',
                        lambda name, improved: SentimentClassifier(name, improved, tiny_encoder(len(tokenizer))))
    frame = build_train_augmentations(3)
    splits = {'train': frame.iloc[:36].copy(), 'validation': frame.iloc[36:45].copy(),
              'test': frame.iloc[45:57].copy()}
    before = {k: v.copy(deep=True) for k, v in splits.items()}
    config = dict(model_name='local', seeds=[42], max_length=64, batch_size=32,
                  epochs=2, learning_rate=0.002, weight_decay=0.01, patience=2)
    torch.set_num_threads(2)
    train_and_eval(config, splits, tokenizer)
    for name in splits:
        pd.testing.assert_frame_equal(splits[name], before[name])
    main = pd.read_csv('results/main_metrics_seeds.csv')
    assert set(main.model) == {'baseline', 'principal', 'improved'}
    edges = pd.read_csv('results/edge_metrics_seeds.csv')
    assert len(edges) == 18
    assert edges[['accuracy', 'macro_f1', 'weighted_f1']].notna().all().all()
    text = (tmp_path / 'results/demo_out.txt').read_text()
    assert text.count('Predicted Sentiment:') == 6
    assert text.count('Management Action:') == 6
    for filename in (tmp_path / 'results').glob('fig*.png'):
        with Image.open(filename) as image:
            assert abs(image.info['dpi'][0] - 300) < 1
    assert len(list((tmp_path / 'results').glob('fig*.png'))) == 5
    saved = torch.load('checkpoints/improved.pt', weights_only=True)
    restored = SentimentClassifier('', True, tiny_encoder(len(tokenizer)))
    restored.load_state_dict(saved['state'])
    p = predict(restored, tokenizer, DEMOS, config)
    assert p.shape == (6, 3)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/experiment.log').write_text('Training completed on CPU.\n')
    script = (tmp_path / 'run_all.sh').read_text().split("python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    subprocess.run([sys.executable, '-c', script], check=True)
    with ZipFile('MFE310TC_Submission_Package.zip') as archive:
        names = archive.namelist()
        assert len(names) == 24
        assert len([n for n in names if n.startswith('results/')]) == 12
        assert not any(n.endswith('.pt') or 'setup_and_run_colab' in n for n in names)



def test_split_isolation(monkeypatch):
    """Confirm train, validation, and test splits have zero overlapping reviews."""
    import datasets
    from src.data import load_splits
    frame = pd.DataFrame({'text': [f'Review {i}' for i in range(100)],
                          'label': [i % 5 for i in range(100)]})
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: SimpleNamespace(to_pandas=lambda: frame.copy()))
    config = dict(dataset_name='local', sample_size=60, data_seed=42)
    splits = load_splits(config)
    assert [len(splits[k]) for k in ['train', 'validation', 'test']] == [42, 9, 9]
    assert not set(splits['train'].text) & set(splits['test'].text)
    assert not set(splits['train'].text) & set(splits['validation'].text)
    assert not set(splits['test'].text) & set(splits['validation'].text)
    for k, subset in splits.items():
        pd.testing.assert_frame_equal(subset, load_splits(config)[k])


def test_default_sample_size(monkeypatch):
    """Verify that default config loads 30,000 deduplicated reviews in a 21k/4.5k/4.5k split."""
    import datasets
    import yaml
    from src.data import load_splits
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'config.yaml').read_text())
    assert config['sample_size'] == 30000
    source = pd.DataFrame({'text': [f'Review {i}' for i in range(50000)],
                           'label': np.arange(50000) % 5})
    source = pd.concat([source, source.iloc[:100]], ignore_index=True)
    calls = []

    def load(name, split):
        calls.append((name, split))
        return SimpleNamespace(to_pandas=lambda: source.copy())

    monkeypatch.setattr(datasets, 'load_dataset', load)
    splits = load_splits(config)
    repeated = load_splits(config)
    all_text = set()
    for name, count in [('train', 21000), ('validation', 4500), ('test', 4500)]:
        frame = splits[name]
        assert len(frame) == count
        assert frame.label.value_counts().to_dict() == dict.fromkeys(range(3), count // 3)
        assert not all_text & set(frame.text)
        all_text.update(frame.text)
        pd.testing.assert_frame_equal(frame, repeated[name])
    assert len(all_text) == 30000
    assert all(split == 'train' for _, split in calls)


@pytest.mark.parametrize('size', [0, -60, 30, 61, 30000.0, True])
def test_invalid_sample_size(size):
    """Check that invalid sample_size values raise a ValueError."""
    from src.data import load_splits
    with pytest.raises(ValueError, match='sample_size'):
        load_splits(dict(sample_size=size))


def test_insufficient_reviews(monkeypatch):
    """Check that datasets with too few unique reviews per class raise a ValueError."""
    import datasets
    from src.data import load_splits
    source = pd.DataFrame({'text': ['same text'] * 100, 'label': [2] * 100})
    monkeypatch.setattr(datasets, 'load_dataset',
                        lambda *a, **kw: SimpleNamespace(to_pandas=lambda: source.copy()))
    with pytest.raises(ValueError, match='unique reviews per class'):
        load_splits(dict(sample_size=60, dataset_name='local', data_seed=42))
