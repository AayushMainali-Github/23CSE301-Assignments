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


def encode_dataframe(data, label_columns, one_hot_columns):
    result = data.copy()
    mappings = {}

    for column in label_columns:
        values = result[column].tolist()
        mappings[column] = label_encoding(values)
        result[column] = [mappings[column][x] for x in values]

    for column in one_hot_columns:
        values = result[column].tolist()
        mappings[column] = one_hot_encoding(values)

        for value in mappings[column]:
            name = column + "_" + str(value)
            result[name] = [1 if x == value else 0 for x in values]

        result = result.drop(columns=[column])

    return result, mappings
