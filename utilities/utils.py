import numpy as np
import pandas as pd 

def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def get_vector_space_info(X):
    dimensionality = X.shape[1]
    number_of_vectors = X.shape[0]
    return dimensionality, number_of_vectors


def calculate_feature_matrix_rank(X):
    return np.linalg.matrix_rank(X)
