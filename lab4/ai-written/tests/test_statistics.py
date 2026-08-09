# GenAI-assisted unit tests: ChatGPT (GPT-5.6 Sol)
import numpy as np, pytest
from dmtoolkit.statistics import mean, variance, standard_deviation, dataset_statistics

def test_mean(): assert np.isclose(mean([1,2,3,4,5]), np.mean([1,2,3,4,5]))
def test_variance(): assert np.isclose(variance([1,2,3,4,5]), np.var([1,2,3,4,5]))
def test_std(): assert np.isclose(standard_deviation([1,2,3,4,5]), np.std([1,2,3,4,5]))
def test_dataset_stats():
    x=np.array([[1,10],[2,20],[3,30],[4,40]], dtype=float)
    means, vars_, stds=dataset_statistics(x.tolist())
    assert np.allclose(means,x.mean(0)) and np.allclose(vars_,x.var(0)) and np.allclose(stds,x.std(0))
def test_empty():
    with pytest.raises(ValueError): mean([])
