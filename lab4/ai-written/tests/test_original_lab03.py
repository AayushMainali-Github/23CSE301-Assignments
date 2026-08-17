# GenAI-assisted tests applied to the student's original Lab 03 functions.
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ORIG=ROOT/"original"
sys.path.insert(0,str(ORIG))
from A2 import label_encoding, one_hot_encoding
from A4 import minkowski_distance
from A7 import my_dot, my_norm
from A8 import my_mean, my_variance, my_std, dataset_stats
from A11 import kmeans
from scipy.spatial.distance import minkowski

def test_original_encoding_shapes():
    vals=["A","B","A"]
    assert set(label_encoding(vals))=={"A","B"}
    oh=one_hot_encoding(vals)
    assert len(oh)==2 and all(len(v)==2 for v in oh.values())

def test_original_minkowski_matches_scipy():
    a,b=[1,2,3],[4,5,6]
    for p in range(1,11): assert np.isclose(minkowski_distance(a,b,p),minkowski(a,b,p))

def test_original_vector_ops_match_numpy():
    a,b=[2,3,4,5],[1,0,2,3]
    assert np.isclose(my_dot(a,b),np.dot(a,b)) and np.isclose(my_norm(a),np.linalg.norm(a))

def test_original_stats_match_numpy():
    x=np.array([[1.,2.],[3.,4.],[5.,6.]])
    means,stds=dataset_stats(x.tolist())
    assert np.allclose(means,x.mean(0)) and np.allclose(stds,x.std(0))
    assert np.isclose(my_mean([1,2,3]),2) and np.isclose(my_variance([1,2,3]),np.var([1,2,3])) and np.isclose(my_std([1,2,3]),np.std([1,2,3]))

def test_original_kmeans_simple_data():
    labels, centroids=kmeans([[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]],2)
    assert len(labels)==6 and len(centroids)==2
