import pandas as pd


def label_encode(values, mapping):
    # convert the values to a list so we can use them more than once
    values = list(values)

    # create a number for every new category
    if len(mapping) == 0:
        for value in values:
            if value not in mapping:
                mapping[value] = len(mapping)

    # replace the original category with its number
    encoded = []
    for value in values:
        encoded.append(mapping[value])

    return encoded, mapping


def one_hot_encode(values, categories, prefix):
    # one hot encoding makes one column for each category
    values = list(values)

    # remember the category order so future data has the same columns
    if len(categories) == 0:
        for value in values:
            if value not in categories:
                categories.append(value)

    # put one in the matching category column and zero in the others
    rows = []
    for value in values:
        row = []
        for category in categories:
            row.append(int(value == category))
        rows.append(row)

    # give the new columns readable names
    columns = []
    for category in categories:
        columns.append(prefix + "_" + str(category))

    return pd.DataFrame(rows, columns=columns), categories


def encode_dataframe(data, label_columns, one_hot_columns):
    # work on a copy so the original table is not changed
    result = data.copy()
    mappings = {}

    # use one number for each value in label-encoded columns
    for column in label_columns:
        result[column], mappings[column] = label_encode(
            result[column], {})

    # replace each categorical column with its one hot columns
    for column in one_hot_columns:
        hot, mappings[column] = one_hot_encode(
            result[column], [], column)
        hot.index = result.index
        result = pd.concat(
            [result.drop(columns=[column]), hot], axis=1)

    return result, mappings
