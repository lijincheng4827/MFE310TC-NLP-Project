# Restaurant Review Sentiment Classification

This project was developed for MFE310TC – Natural Language Processing and Deep Learning, Track 1. It compares three approaches to classifying restaurant reviews as negative, neutral or positive, with a separate evaluation set covering Middle Eastern dining and examples that include slang, emojis, sarcasm and changes in opinion within a review.

## Experiment setup

The experiments were conducted on Google Colab Pro with an NVIDIA A100-SXM4-80GB GPU, Ubuntu Linux, Python 3.13 and CUDA 13.0. Each model is trained with seeds 42, 123 and 2026 so that its average performance and variation across runs can be compared, while each neural model is trained for up to five epochs and stops early if validation Macro-F1 does not improve for two consecutive epochs.

The experiment saves metric tables and five 300 DPI figures in `results/`, records training details in `logs/experiment.log`, and stores model checkpoints in `checkpoints/`. It also creates `MFE310TC_Submission_Package.zip` containing the code and experiment outputs, with the checkpoints left out of the archive.

## Data

The main dataset contains 30,000 unique Yelp reviews, with equal numbers of negative, neutral and positive examples. Reviews with one or two stars are labelled negative, three-star reviews are labelled neutral, and reviews with four or five stars are labelled positive. Duplicate texts are removed before sampling from the original Yelp training partition, after which the sample is divided into training, validation and test sets while keeping the class proportions the same.

| Split | Reviews | Per sentiment class |
| --- | ---: | ---: |
| Train | 21,000 | 7,000 |
| Validation | 4,500 | 1,500 |
| Test | 4,500 | 1,500 |

Sampling uses `data_seed: 42`, and all models use these same splits across their training runs so that their scores can be compared on the same reviews. The improved model receives another 180 dining examples during training, including 30 longer reviews with contrasting opinions, which increases its training set to 21,180 reviews without changing the validation or test sets.

Because the main Yelp sample covers businesses in general, the project also uses a manually written set of 240 reviews focused on Middle Eastern dining. This set contains six categories of 40 reviews each, with all three sentiment labels represented in every category, and is used to examine how the models handle the language patterns chosen for this project.

## Models

The `baseline` model uses TF-IDF features from individual words and word pairs with logistic regression, providing a starting point for comparison with the two neural models. The `principal` model uses DistilBERT with its first-token representation passed to a classifier and is trained with cross-entropy loss.

The `improved` model, DCPA-DistilBERT, adds text normalization and attention-based pooling to help it handle informal language and reviews whose final opinion differs from their opening sentences. Its preprocessing converts selected emojis, slang and spelling variants into more familiar expressions, and adds a negative summary for selected patterns where praise is followed by an adverse event. During training, it combines label-smoothed cross-entropy loss with focal loss using weights of 80% and 20%, respectively.

For long reviews, the improved tokenizer keeps text from both the beginning and the end so that truncation is less likely to remove a closing complaint. The model learns token attention weights and gives extra weight to the final contrast clause when a review has more than 40 content tokens and a visible contrast marker. When there is no contrast marker, the pooling calculation uses the final quarter of the review as a fallback. The checkpoint for each run is selected using validation Macro-F1, after which the model is evaluated on the main test set and the separate dining review set.

## Results and demo

The experiment writes scores for individual runs to `main_metrics_seeds.csv` and `edge_metrics_seeds.csv`, then reports their means and sample standard deviations in `main_metrics.csv` and `edge_metrics.csv`. The tables include accuracy, Macro-F1 and Weighted-F1; Macro-F1 averages over the classes present in the reference labels. Incorrect predictions are saved in `error_cases.csv`, while `train_history.csv` records the training progress of each neural model across epochs.

The five figures show the data distribution, model comparison, confusion matrices, performance on the dining review categories and attention weights. Confusion matrices and attention plots use seed 42, and subword attention weights are added together to display a single weight for each word.

The six-review demo shows predicted sentiment probabilities for each complete review and for separate food and service clauses, then suggests a management action based on those predictions. After training, the demo can be repeated with `python -m src.demo`, and its text output is saved in `results/demo_out.txt`.

## Project files

Data loading, sampling and text normalization are implemented in `src/data.py`, while `src/models.py` defines the neural classifiers and loss function. The training and plotting workflow is in `src/train.py`, the dining review set and evaluation metrics are in `src/edge_eval.py`, and the customer feedback demo is in `src/demo.py`.

Training settings, including sample size, seeds, batch size and learning rate, are stored in `config.yaml`. The existing tests can be run with `python -m pytest -q` and cover preprocessing, data splitting, model pooling and an end-to-end run with a small local model.
