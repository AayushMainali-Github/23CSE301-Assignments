"""Preprocessing utilities. GenAI-assisted: ChatGPT (GPT-5.6 Sol)."""
import pandas as pd

def label_encode(values):
    values = list(values)
    if not values:
        raise ValueError("Input values cannot be empty.")
    mapping, encoded = {}, []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return encoded, mapping

def one_hot_encode(values, prefix="category"):
    values = list(values)
    if not values:
        raise ValueError("Input values cannot be empty.")
    categories = []
    for value in values:
        if value not in categories:
            categories.append(value)
    mapping = {cat: [1 if i == j else 0 for j in range(len(categories))]
               for i, cat in enumerate(categories)}
    rows = [mapping[v] for v in values]
    columns = [f"{prefix}_{cat}" for cat in categories]
    return pd.DataFrame(rows, columns=columns), mapping

def encode_dataframe(data, label_columns=None, one_hot_columns=None, drop_columns=None):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    label_columns = label_columns or []
    one_hot_columns = one_hot_columns or []
    drop_columns = drop_columns or []
    out = data.copy()
    mappings = {}
    for column in label_columns:
        if column not in out.columns:
            raise KeyError(column)
        if out[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        vals, mp = label_encode(out[column].tolist())
        out[column] = vals
        mappings[column] = {"type": "label", "mapping": mp}
    for column in one_hot_columns:
        if column not in out.columns:
            raise KeyError(column)
        if out[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        hot, mp = one_hot_encode(out[column].tolist(), prefix=column)
        hot.index = out.index
        out = pd.concat([out.drop(columns=[column]), hot], axis=1)
        mappings[column] = {"type": "one_hot", "mapping": mp}
    for column in drop_columns:
        if column in out.columns:
            out = out.drop(columns=[column])
    return out, mappings
