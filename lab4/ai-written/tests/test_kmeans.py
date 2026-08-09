# GenAI-assisted unit tests: ChatGPT (GPT-5.6 Sol)
import numpy as np, pytest
from dmtoolkit.kmeans import kmeans_ai

DATA=[[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]]

def test_returns_expected_shapes():
    r=kmeans_ai(DATA,2)
    assert len(r["centroids"])==2 and len(r["labels"])==6

def test_nonnegative_inertia(): assert kmeans_ai(DATA,2)["inertia"] >= 0

def test_reproducible():
    a,b=kmeans_ai(DATA,2),kmeans_ai(DATA,2)
    assert np.allclose(a["centroids"],b["centroids"])

def test_invalid_k():
    with pytest.raises(ValueError): kmeans_ai(DATA,100)
