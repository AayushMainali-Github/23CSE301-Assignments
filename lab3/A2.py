def label_encoding(datalist):
    unq = list(set(datalist))
    mapping = {}
    for i in range(len(unq)):
        mapping[unq[i]] = i
    return mapping


def one_hot_encoding(datalist):
    unique = list(set(datalist))
    n = len(unique)
    mapping = {}

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        mapping[unique[i]] = row

    return mapping
