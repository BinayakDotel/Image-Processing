import math

prime = []
value = 6008514751

def checkPrime(number):
    flag = True

    if number == 0 or number == 1:
        return False
    
    for i in range(2,number):
        if number%i == 0:
            flag = False
            break

    return flag

m = math.ceil(math.sqrt(value))
for num in range(2, m+1):
    if checkPrime(num):
        if value%num==0:
            prime.append(num)

print(prime)
#last_index = len(prime)
#rint(f"Largest is :: {prime[last_index-1]}")
    

