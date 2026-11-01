# Restaurant Review Sentiment Classification

MFE310TC – Natural Language Processing and Deep Learning, Track 1.
Sentiment classification with Middle Eastern restaurant examples.

## Run

The experiments were conducted on Google Colab Pro using an NVIDIA A100-SXM4-80GB GPU (Ubuntu Linux, Python 3.13, CUDA 13.0). To reproduce the complete training and evaluation pipeline, run:

```bash
python -m pip install -r requirements.txt
bash run_all.sh
```

The complete experiment trains three models for seeds 42, 123 and 2026,
writes tables and five 300 DPI figures to `results/`, and creates
`MFE310TC_Submission_Package.zip`. Training details go to `logs/experiment.log`.
Checkpoints stay in `checkpoints/` and are excluded from the zip.
Run `python -m src.demo` to repeat the six-review demo after training.

The default experiment uses 30,000 reviews and up to five epochs for each
neural model and seed. Early stopping uses two epochs without validation
improvement. All models use the same `sample_size` setting.

## Data and models

A fixed, balanced sample of 30,000 unique Yelp reviews uses a 70/15/15
stratified train/validation/test split. One- and two-star reviews are negative,
three-star reviews neutral, and four- and five-star reviews positive.

| Split | Reviews | Per sentiment class |
| --- | ---: | ---: |
| Train | 21,000 | 7,000 |
| Validation | 4,500 | 1,500 |
| Test | 4,500 | 1,500 |

The same splits are used for every model and training seed. Sampling uses
`data_seed: 42` independently of the training seeds. Exact duplicate review
texts are removed before sampling, and the original Yelp test partition is
not used. Each improved training run has 21,180 reviews after augmentation.
Split sizes and class counts are recorded in the experiment log. The main sample
covers Yelp businesses generally; the separate 240-review suite focuses on
Middle Eastern dining. Each of its six categories has 40 reviews with all
three sentiment classes. The suite is a fixed, manually designed evaluation set.

- `baseline`: word and bigram TF-IDF with logistic regression.
- `principal`: DistilBERT first-token pooling with cross-entropy loss.
- `improved`: DCPA-DistilBERT, with emoji and slang normalization, an irony
  summary when praise accompanies an adverse event, learned token attention,
  and a residual pool emphasizing the final contrast clause in long reviews.
  Training adds 180 separate dining examples, including 30 longer contrasts,
  and combines label-smoothed cross-entropy (80%) with focal loss (20%).

Improved tokenization preserves both ends of long reviews. A final-quarter
pool is available when no contrast marker occurs. The contrast residual gate
activates only for reviews longer than 40 content tokens with a visible final
contrast clause. Augmentations never enter validation or test data. Epoch
selection uses validation Macro-F1 only; the edge suite is evaluated afterward.

## Results

`main_metrics_seeds.csv` and `edge_metrics_seeds.csv` contain individual runs.
`main_metrics.csv` and `edge_metrics.csv` report means and sample standard
deviations. Macro-F1 uses the classes present in the gold labels; Weighted-F1
is also reported. `error_cases.csv` contains incorrect main and edge predictions,
and `train_history.csv` records each neural epoch. Confusion matrices and
attention plots use seed 42. WordPiece weights are summed into words for display.

The demo reports model probabilities for whole reviews and separate cuisine
and service clauses. Actions follow predicted sentiment and aspects. The text
normalizer includes explicit domain rules, so comparisons measure the complete
preprocessing and training strategies as well as the pooling architecture.

## Files

`src/data.py` handles data and normalization; `src/models.py` defines the neural
models; `src/train.py` runs training and plots; `src/edge_eval.py` defines the
restaurant suite and metrics; `src/demo.py` runs the customer feedback examples.
Edit `config.yaml` to change training settings. Run `python -m pytest -q` for tests.
