print("No of rows = ")
rows = int(input())
print("No of columns = ")
cols = int(input())

A = []
print("Enter matrix elements")
for i in range(rows):
    r = list(map(int, input().split()))
    A.append(r)

B = []
for j in range(cols):
    r = []
    for i in range(rows):
        r.append(A[i][j])
    B.append(r)

print("Transpose: ")
for r in B:
    print(r)