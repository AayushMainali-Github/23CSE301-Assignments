print("Enter the string")
st = input()

vowCnt = 0
conCnt = 0

for a in st:
    if a in "aeiouAEIOU":
        vowCnt = vowCnt + 1
    else:
        conCnt = conCnt + 1

print(f"Vowel Count = {vowCnt}\nConsonant Count = {conCnt}")
