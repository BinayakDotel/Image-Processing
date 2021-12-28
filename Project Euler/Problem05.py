'''
2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.

What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
'''
max_index = 20
initial = max_index
found = False

while not found:
  for i in range(1, max_index+1):
    if (initial % i) != 0:
      found = False
      initial = initial + max_index
      break
    else:
      found = True

print(initial)
