def sort_key(item):
    # sort first by distance and then by original training index
    return item[0], item[1]


def bubble_sort(items):
    items = list(items)

    # bubble sort
    # the largest remaining item moves to the end each round
    for end in range(len(items) - 1, 0, -1):
        for i in range(end):
            if sort_key(items[i]) > sort_key(items[i + 1]):
                items[i], items[i + 1] = items[i + 1], items[i]

    return items


def insertion_sort(items):
    result = []

    # insertion sort
    # put each new item into its correct place in the result
    for item in items:
        position = len(result)

        while position > 0:
            if sort_key(item) >= sort_key(result[position - 1]):
                break
            position = position - 1

        result.insert(position, item)

    return result


def merge_sort(items):
    items = list(items)

    # a one-item list is already sorted
    if len(items) < 2:
        return items

    # split the list and sort both smaller lists
    middle = len(items) // 2
    left = merge_sort(items[:middle])
    right = merge_sort(items[middle:])
    result = []

    # take the smaller front item from the two lists
    while len(left) > 0 and len(right) > 0:
        if sort_key(left[0]) <= sort_key(right[0]):
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))

    return result + left + right


def sort_distances(items, method):
    # select one of the three sorting algorithms
    if method == "bubble":
        return bubble_sort(items)

    if method == "insertion":
        return insertion_sort(items)

    if method == "merge":
        return merge_sort(items)

    raise ValueError("use bubble, insertion, or merge")
