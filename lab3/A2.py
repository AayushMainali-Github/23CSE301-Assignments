import pandas as pd

def label_encoding(datalist):
    unq = list(set(datalist))
    n = len(unq)
    mapping = {}
    for i in range(0,n):
        mapping[unq[i]] = i
    return mapping
        
    
    
    
def one_hot_encoding(datalist):
    unique = list(set(datalist))
    int n = len(datalist)
    matrix = []
    for i in range(0,n):
        row = []
        for j in range (0,n):
            if i == j: 
                row[j] = 1
            else: 
                row[j] = 0
        matrix[i] = row

    mapping = {}
    for i in range(0,n):
        mapping[unique[i]] = matrix[i]  

    return mapping  
