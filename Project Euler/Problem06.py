'''The sum of the squares of the first ten natural numbers is,

The square of the sum of the first ten natural numbers is,

Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is .

Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.'''

import math

max_value = 100
sum_of_square = 1
square_of_sum = 1
for i in range(2, max_value+1):
  sum_of_square = sum_of_square + (i*i)

  square_of_sum = square_of_sum + i

square_of_sum = math.pow(square_of_sum, 2)

print(square_of_sum - sum_of_square)