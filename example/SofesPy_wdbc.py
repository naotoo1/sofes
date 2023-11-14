"""
Prototype-based global feature selection example using the wdbc dataset
"""

import numpy as np
from sofes import (
    dataset,
    SofesPy,
    seed_everything,
    proto_initializer,
    LVQ
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

    # Initialise prototypes with means
    proto_init = proto_initializer.get_Kmeans_prototypes(
            input_data=input_data,num_cluster=num_classes
            ).Prototypes

    #latent dimension
    latent_dim = input_data.shape[1]

    # initialise a prototype-based induction learner
    learner = LVQ.MRSLVQ.value

    # set up the prototype feature selection
    train = SofesPy(
        input_data=input_data,
        labels=labels,
        model_name=learner,
        latent_dim=latent_dim,
        sigma=1,
        num_classes=num_classes,
        init_prototypes=proto_init,
        init_matrix = None,
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

    # Summary of significant features
    print(
        'significant_features=',
        summary.significant,
    )

    # Summary of insignificant features
    print(
        'insignificant_features=',
        summary.insignificant,
    )
