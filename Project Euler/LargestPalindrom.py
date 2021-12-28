'''A palindromic number reads the same both ways. 
    The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.

Find the largest palindrome made from the product of two 3-digit numbers.'''

max_value = 999
largest = 0
for i in range(100, max_value+1):
  for j in range(100, max_value+1):
    prod = i * j
    prodS = str(prod)
    reverse = prodS[::-1]
    if prodS == reverse:
      if prod > largest:
        largest = prod
        print(f"Found::{i}*{j} => {prod}")

print(largest)