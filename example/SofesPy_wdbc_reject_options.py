"""
Prototype-based local feature selection with reject options example using the wdbc dataset
"""

import numpy as np
from sofes import (
    dataset,
    LVQ,
    SofesPy,
    seed_everything,
    get_rejection_summary
)

if __name__ == "__main__":
    # Reproducibility
    seed_everything(seed=4)

    # Dataset
    train_data = dataset.DATA(random=4)

    # Input features of the dataset
    input_data = train_data.breast_cancer.input_data

    # Targets of the input features
    labels = train_data.breast_cancer.labels

    # Number of classes
    num_classes = len(np.unique(labels))

    # latent dimension
    latent_dim = input_data.shape[1]

    # initialise a prototype-based induction learner
    learner = LVQ.LMRSLVQ.value

    # Setup the prototype feature selection
    train = SofesPy(
        input_data=input_data,
        labels=labels,
        model_name=learner,
        latent_dim=latent_dim,
        sigma=1,
        num_classes=num_classes,
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

    print(
        "----------------------With reject_strategy----------------------------"
    )

    # Summary of significant features with rejection strategy
    print(
        'significant_features=',
        rejected_strategy.significant
    )

    # Summary of the insignificant features with rejection strategy
    print(
        'insignificant_features=',
        list(set(rejected_strategy.insignificant) - set(rejected_strategy.tentative))
    )

    # Summary of the tentative features with rejection strategy
    print(
        'tentative_features=',
        rejected_strategy.tentative
    )
