# -*- coding:utf-8 -*-
"""
题目描述
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
并保证奇数和奇数，偶数和偶数之间的相对位置不变。
"""

class Solution:
    def reOrderArray(self, array):
        # write code here
        res_odd = []
        res_even = []
        for i in range(len(array)):
            
            if array[i]%2:
                print(array[i])
                res_odd.append(array[i])
            else:
                res_even.append(array[i])

        res_even.sort()
        res_odd.sort()
        return res_odd+res_even

s = Solution()
print(s.reOrderArray([1,2,3,4,5,6,7]))