import numpy as np
from sofes import dataset


# load the breast cancer dataset
data = dataset.DATA()
input_data =data.breast_cancer.input_data
labels = data.breast_cancer.labels


def test_breast_cancer():
    assert len(input_data)==569
    assert len(labels)==569
    assert len(np.unique(labels))==2
    assert len(input_data.shape[1])==30





