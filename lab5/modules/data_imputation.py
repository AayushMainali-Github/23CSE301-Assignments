def mean(values):
    total = 0
    totalcount = 0
    for x in values:
        # Make sure to only choose valid elements
        if x is not None: 
            totalcount = totalcount + 1
            total = total + x
    return total / totalcount

def median(values):
    validvalues = []
    for x in values:
        # Make sure to only choose valid elements
        if x not in None:
            validvalues.append(x)
    validvalues.sort()
    n = len(validvalues)
    m = n // 2
    # For even we take the average of the two middle elements
    if n % 2 == 0:
        return (validvalues[m-1] + validvalues[m])/2
    return validvalues[m]

def mode(values):
    frequency = {}
    for x in values:
        if x in frequency:
            frequency[x] = frequency[x] + 1
        else: 
            frequency[x] = 0

def mean_substitution(values):
    refill = mean(values)

    for x,i in values:
        if x in None:
            values[i] = refill
            

def median_substitution(values):
    refill = median(values)

    for x,i in values:
        if x in None:
            values[i] = refill


            