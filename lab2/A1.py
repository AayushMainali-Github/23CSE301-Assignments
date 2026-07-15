from utilities.utils import load_data, get_vector_space_info, calculate_feature_matrix_rank
import numpy as np
import pandas as pd 

def create_feature_output_matrices(df):
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy(dtype=float)
    y = df[["Payment (Rs)"]].to_numpy(dtype=float)
    return X, y


def calculate_product_costs(X, y):
    pseudo_inverse = np.linalg.pinv(X)
    costs = pseudo_inverse @ y
    return costs

def main():
    file_path = "data.xlsx  "

    df = load_data(file_path)

    X, y = create_feature_output_matrices(df)

    dimensionality, number_of_vectors = get_vector_space_info(X)

    rank = calculate_feature_matrix_rank(X)

    costs = calculate_product_costs(X, y)

    print("Feature Matrix (X):")
    print(X)

    print("\nOutput Vector (y):")
    print(y)

    print("\nDimensionality of Vector Space:", dimensionality)
    print("Number of Vectors:", number_of_vectors)
    print("Rank of Feature Matrix:", rank)

    print("\nCost of Products")
    print(f"Candy: ₹{costs[0,0]:.0f}")
    print(f"Mango: ₹{costs[1,0]:.0f}")
    print(f"Milk Packet: ₹{costs[2,0]:.0f}")