import numpy as np
from sofes import (dataset,proto_initializer)




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


def test_prototype_initializer():
    mean_initializer =proto_initializer.get_Kmeans_prototypes(
        input_data=input_data,num_cluster=2
        ).Prototypes
     
    assert len(mean_initializer.shape[1])==6
    assert len(mean_initializer)==2