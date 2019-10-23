def partition(array,left,right):
    if left >= right:
        return 
    privot = left
    i = left
    j = right
    while i < j:
        while array[j] > array[privot] and j > i:
            j -= 1
        while array[i] <= array[privot] and i < j:
            i += 1
        array[j],array[i] = array[i],array[j]
    array[privot],array[i] = array[i],array[privot]
    privot = j
    partition(array,left, privot-1)
    partition(array, privot+1, right)
    
def partition_sort(array):
    if not array: return
    partition(array,0,len(array)-1)

if __name__ == "__main__":
    array = [2,5,6,1,8,2,2,2,2,9]
    partition_sort(array)
    print(array)