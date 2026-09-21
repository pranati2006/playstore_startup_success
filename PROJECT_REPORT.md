# Project Report: Semantic Startup App Success Analysis

## Abstract

This project proposes a machine-learning system for evaluating a proposed mobile application by comparing it with semantically similar Google Play applications. Public app metadata is collected from Google Play, while full app descriptions are transformed into dense semantic embeddings using a Sentence Transformer. For a proposed startup idea, cosine similarity identifies the closest applications in the existing market. Their installs, reviews, ratings and monetization patterns are summarized into competitor-market features. Historical applications are labelled successful or unsuccessful using a category-aware score based on install and review percentiles. Logistic Regression, Random Forest and Extra Trees classifiers are trained and compared. The best model is selected using ROC-AUC with F1-score as a tie-breaker. The final system displays a success-profile score and the closest comparable applications through a Streamlit interface.

## Problem Statement

Startup teams often need an early indication of how crowded a proposed app market is and how similar applications perform. Raw category labels are too broad: two apps can be in the same category while solving very different problems. This project therefore uses semantic similarity over full app descriptions before applying machine learning.

## Objectives

- Collect current Google Play app metadata and full descriptions.
- Convert descriptions into semantic vector embeddings.
- Retrieve top-K semantically similar applications.
- Engineer market-level features from comparable apps.
- Define historical success using installs and reviews.
- Train and compare multiple classification algorithms.
- Prevent target leakage.
- Return one selected-model score plus comparable-app evidence.
- Provide an interactive interface.

## Success Definition

For each historical app:

```text
InstallPercentile = rank(log(1 + installs))
ReviewPercentile  = rank(log(1 + reviews))

SuccessScore =
    0.70 * InstallPercentile
  + 0.30 * ReviewPercentile

Successful = 1 when SuccessScore >= 0.75
```

Percentiles are calculated inside the app genre when the genre has enough observations; otherwise global percentiles are used.

## Semantic Retrieval

Full descriptions are encoded with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are L2-normalized. The nearest apps are retrieved with cosine distance.

## Features

Candidate-side features:

- description embedding
- genre
- content rating
- free/paid
- price
- in-app purchases
- ads
- description length
- summary length

Semantic-market features:

- mean similarity
- median comparable installs
- median comparable reviews
- mean comparable rating
- free-app ratio
- IAP ratio
- ads ratio
- successful-neighbor ratio
- close-competitor count

The candidate app's own installs and reviews are never used as predictors.

## Models

The project compares:

1. Logistic Regression
2. Random Forest
3. Extra Trees

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Leakage-Safe Evaluation

The dataset is split into train and test sets before evaluation market features are constructed.

- A training app may use only other training apps as semantic references.
- A test app may use only training apps as semantic references.
- The test app's installs, reviews and success label therefore cannot enter its market-feature reference pool.
- After model selection, the chosen model is retrained on all historical data for deployment.

## Final Output

For a startup idea, the application returns:

- success-profile score
- selected ML model
- semantic market similarity
- median competitor installs
- median competitor reviews
- average competitor rating
- successful-neighbor ratio
- close-competitor count
- table of nearest Google Play applications

## Limitations

This is not a causal prediction of business success. Current installs and reviews are affected by app age, advertising, brand strength, geography, prior user base and survivorship. A one-time Play Store snapshot also cannot directly learn future growth.

The strongest future improvement is to collect repeated snapshots over time and predict future install/review growth from information available at an earlier date.

## Conclusion

The project combines NLP semantic embeddings with structured machine learning to create a market-aware app analysis system. Its main advantage over a normal Play Store classifier is that it explicitly identifies comparable applications before estimating how closely the proposed idea matches historically successful app profiles.
