import pandas as pd


def unique_values(values):
    unique = []

    for value in values:
        if value not in unique:
            unique.append(value)

    return unique


def label_encoding(datalist):
    # give every different value one number
    unique = unique_values(datalist)
    mapping = {}

    for i in range(len(unique)):
        mapping[unique[i]] = i

    return mapping


def one_hot_encoding(datalist):
    # make one binary list for every different value
    unique = unique_values(datalist)
    mapping = {}

    for i in range(len(unique)):
        row = []
        for j in range(len(unique)):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        mapping[unique[i]] = row

    return mapping


def label_encode(values, mapping):
    values = list(values)

    if len(mapping) == 0:
        mapping = label_encoding(values)

    encoded = []
    for value in values:
        encoded.append(mapping[value])

    return encoded, mapping


def one_hot_encode(values, categories, prefix):
    values = list(values)

    if len(categories) == 0:
        categories = unique_values(values)

    rows = []
    for value in values:
        row = []
        for category in categories:
            row.append(int(value == category))
        rows.append(row)

    columns = []
    for category in categories:
        columns.append(prefix + "_" + str(category))

    return pd.DataFrame(rows, columns=columns), categories


def encode_dataframe(data, label_columns, one_hot_columns):
    result = data.copy()
    mappings = {}

    for column in label_columns:
        result[column], mappings[column] = label_encode(
            result[column], {})

    for column in one_hot_columns:
        hot, mappings[column] = one_hot_encode(
            result[column], [], column)
        hot.index = result.index
        result = pd.concat(
            [result.drop(columns=[column]), hot], axis=1)

    return result, mappings
