import random
import statistics

nums = []
for i in range(100):
    nums.append(random.randint(100, 150))

mean = statistics.mean(nums)
median = statistics.median(nums)
mode = statistics.mode(nums)

print("Generated Numbers")
print(nums)
print(f"Mean = {mean}")
print(f"Median = {median}")
print(f"Mode = {mode}")