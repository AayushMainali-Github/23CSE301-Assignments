# GenAI-assisted unit tests: ChatGPT (GPT-5.6 Sol)
import numpy as np, pytest
from scipy.spatial.distance import minkowski
from dmtoolkit.distances import minkowski_distance, manhattan_distance, euclidean_distance

def test_manhattan(): assert np.isclose(manhattan_distance([1,2,3],[4,5,6]),9)
def test_euclidean(): assert np.isclose(euclidean_distance([0,0],[3,4]),5)
def test_zero_distance(): assert minkowski_distance([2,4],[2,4],2) == 0

def test_minkowski_against_scipy():
    a,b=[2,3,7,9],[1,8,4,5]
    for p in range(1,11): assert np.isclose(minkowski_distance(a,b,p), minkowski(a,b,p))

def test_bad_dimensions():
    with pytest.raises(ValueError): minkowski_distance([1],[1,2],2)

def test_bad_p():
    with pytest.raises(ValueError): minkowski_distance([1],[2],0)
