print("No of rows in A = ")
ra = int(input())
print("No of columns in A = ")
ca = int(input())
print("No of rows in B = ")
rb = int(input())
print("No of columns in B = ")
cb = int(input())

A = []
print("Enter elements of Matrix A")
for i in range(ra):
    r = list(map(int, input().split()))
    A.append(r)



B = []
print("Enter elements of Matrix B")
for i in range(rb):
    r = list(map(int, input().split()))
    B.append(r)

if ca != rb:
    print("Error: Matrices cannot be multiplied")
else:
    C = []
    for i in range(ra):
        r = []
        for j in range(cb):
            total = 0
            for k in range(ca):
                total = total + A[i][k] * B[k][j]
            r.append(total)
        C.append(r)
    print("Product: ")
    for r in C:
        print(r)