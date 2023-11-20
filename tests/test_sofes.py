#!/usr/bin/env python

import numpy as np
from sofes import (
    dataset,
    LVQ,
    SofesPy,
    seed_everything,
    get_rejection_summary
)


# Reproducibility
seed_everything(seed=4)

# Dataset
train_data = dataset.DATA(random=4)

# Input features of the dataset
input_data = train_data.breast_cancer.input_data

# Targets of the input features
labels = train_data.breast_cancer.labels


def test_sofes_rejection_strategy():
 
    # Setup the prototype feature selection
    train = SofesPy(
        input_data=input_data,
        labels=labels,
        model_name=LVQ.LMRSLVQ.value,
        latent_dim=input_data.shape[1],
        sigma=1,
        num_classes=len(np.unique(labels)),
        init_prototypes=None,
        init_matrix=None,
        num_prototypes=1,
        eval_type='mv',
        significance=False,
        evaluation_metric='accuracy',
        perturbation_ratio=0.1,
        perturbation_distribution="uniform",
        epsilon=0.05,
        norm_ord='fro',
        termination='metric',
        verbose=1,
        max_epochs=100,
        regularization=0.0,
        random_state=4,
    )

    # Summary of SofesPy
    summary = train.summary_results

    # Setup rejection strategy
    rejected_strategy = get_rejection_summary(
        significant=summary.significant.features,
        insignificant=summary.insignificant.features,
        significant_hit=summary.significant.hits,
        insignificant_hit=summary.insignificant.hits,
        reject_options=True,
        vis=True,
    )

    significant_features=rejected_strategy.significant
    insignificant_features=list(set(rejected_strategy.insignificant) -\
                                set(rejected_strategy.tentative))

    tentative_features=rejected_strategy.tentative

    feature_space = significant_features + insignificant_features + tentative_features

    assert input_data.shape[1] == len(feature_space)