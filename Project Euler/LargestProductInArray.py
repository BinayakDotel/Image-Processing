def solution(inputArray):
    largest = 0
    for i in range(0, len(inputArray)-1):
        product = inputArray[i] * inputArray[i+1]
        print(f"Compare {inputArray[i]}, {inputArray[i+1]} ==> prod:{product} larg:{largest}")

        if i==0:
            largest = product
        if product > largest:
            print(f"Set larg{largest} == P{product}")
            largest = product
            continue
        
    return largest


if __name__=="__main__":
    print(solution([-23, 4, -3, 8, -12]))