
import numpy as np
from sofes import (mutate_labels,mutated_validation)



input_data =np.array(
        [[1,2,3,4,1,3],
        [1,0,3,4,1,3],
        [1,2,3,9,1,3],
        [6,2,5,4,1,3],
        [1,0,3,2,1,7],
        [9,2,6,4,8,3],
        [2,2,1,4,7,3],
        [2,2,0,4,7,3],
        [0,2,1,4,7,3],
        [2,2,3,4,7,3],
        [1,2,1,4,7,3],
        [1,5,1,4,1,3],
        [3,1,1,2,6,1],
        [9,1,3,1,6,4]])

input_lables =np.array(
    [0,1,1,1,1,0,0,0,1,0,1,1,0,1]
    )



def test_mutation_type():
    mutation_types=mutate_labels.MutationType
    assert mutation_types.SWAPNEXTLABEL == "next"
    assert mutation_types.SWAPLABEL == "unique"

def test_PerturbationDistribution():
    perturbation_type = mutate_labels.PerturbationDistribution
    assert perturbation_type.BALANCEDCLASSAWARE.value == "balanced"
    assert perturbation_type.UNIFORMCLASSAWARE.value=="uniform"
    assert perturbation_type.GLOBAL.value=="global"


def test_target_info():
    perturbation_ratio=0.2
    target_info = mutate_labels.get_target_info(
        labels=input_lables,
        perturbation_ratio=perturbation_ratio
    )

    assert len(target_info.label_indices)== len(input_lables)
    assert target_info.mutation_value == 0.2
    assert len(target_info.unique_labels) == len(np.unique(input_lables))


def test_mutated_validation():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution='uniform'
        )
    
    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1

def test_evaluationmetric():
    predicted_labels= [0,1,1,1,0,0]
    original_labels = [0,1,1,0,1,0]
    metric_score = mutated_validation.get_metric_score(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric='accuracy'
        )
    
    assert metric_score.metric_score == 0.8
    assert metric_score.original_labels == original_labels
    assert metric_score.predicted_labels ==predicted_labels




