# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        length = len(array[0])-1
        height = len(array)-1
        m= 0
        n = height
        while n >=0 and m <= length:
            
            if array[n][m] < target:
                m += 1
            elif array[n][m] > target:
                n -= 1
                m = 0
            else:
                return True
        return False
    
# m = [[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]
s = Solution()
# print(s.Find(15,m))
# print(s.Find(5,[[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]))
print(s.Fibonacci(20))
a = " "
a.split()