"""Descriptive statistics. GenAI-assisted: ChatGPT (GPT-5.6 Sol)."""
import math

def mean(values):
    values = [float(v) for v in values]
    if not values:
        raise ValueError("Values cannot be empty.")
    return sum(values)/len(values)

def variance(values, ddof=0):
    values = [float(v) for v in values]
    if not values:
        raise ValueError("Values cannot be empty.")
    if ddof < 0 or len(values)-ddof <= 0:
        raise ValueError("Invalid ddof.")
    m = mean(values)
    return sum((v-m)**2 for v in values)/(len(values)-ddof)

def standard_deviation(values, ddof=0):
    return math.sqrt(variance(values, ddof))

def dataset_statistics(matrix, ddof=0):
    rows = [list(r) for r in matrix]
    if not rows or not rows[0]:
        raise ValueError("Matrix cannot be empty.")
    ncols = len(rows[0])
    if any(len(r) != ncols for r in rows):
        raise ValueError("All rows must have equal length.")
    means, variances, stds = [], [], []
    for j in range(ncols):
        col = [r[j] for r in rows]
        means.append(mean(col)); variances.append(variance(col, ddof)); stds.append(standard_deviation(col, ddof))
    return means, variances, stds
