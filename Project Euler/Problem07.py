'''
    By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.

    What is the 10 001st prime number?
'''

#Stores all the prime numbers
all_data = []

#It is the desired position of prime number that we want from the all_data list
desire = 10001
found = False
index = 1

#Check prime number
def CheckPrime(number):
    if number == 0 or number == 1:
        return False

    for value in range(2, number):
        if number%value == 0:
            return False

    return True

while not found:
    #Check if the the list length exeeded the desired position or not
    if len(all_data)+1 <= desire:
        if CheckPrime(index):
            all_data.append(index)
        index = index+1

    else:
        found = True
        break
  
print(all_data)
print(all_data[desire-1])
