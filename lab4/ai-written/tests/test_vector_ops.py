# GenAI-assisted unit tests: ChatGPT (GPT-5.6 Sol)
import numpy as np, pytest
from dmtoolkit.vector_ops import dot_product, euclidean_norm

def test_dot_product(): assert np.isclose(dot_product([2,3,4,5],[1,0,2,3]), np.dot([2,3,4,5],[1,0,2,3]))
def test_norm(): assert np.isclose(euclidean_norm([3,4]), 5)
def test_zero_norm(): assert euclidean_norm([0,0,0]) == 0

def test_bad_dimensions():
    with pytest.raises(ValueError): dot_product([1,2],[1])
