def merge(left,right):
    new_array = []
    l ,r = 0,0
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            new_array.append(left[l])
            l += 1
        else:
            new_array.append(right[r])
            r += 1
    if l == len(left):
        new_array += right[r:]
        # for i in right[r:]:
        #     new_array.append(i)
    else:
        new_array += left[l:]
        # for i in left[l:]:
        #     new_array.append(i)
    return new_array
def merge_sort(array):
    if len(array) <= 1:
        return array
    mid = len(array)//2
    l = merge_sort(array[:mid])
    r = merge_sort(array[mid:])
    return merge(l,r)

if __name__ == "__main__":
    array = [2,1,3,4,7]
    print(merge_sort(array))
        