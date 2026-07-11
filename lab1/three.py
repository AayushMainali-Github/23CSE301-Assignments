print("Enter first list = ")
l1 = list(map(int, input().split()))
print("Enter second list = ")
l2 = list(map(int, input().split()))
cnt = 0
for a in l1:
    if a in l2:
        cnt = cnt + 1
        l2.remove(a)

print(f"Total Common Elements = {cnt}")