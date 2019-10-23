# -*- coding:utf-8 -*-
"""
题目描述
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
https://blog.csdn.net/u010005281/article/details/79851154
"""

class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        while n&0xffffffff:
            count += 1
            n &= (n-1)
        return count

s = Solution()
print(s.NumberOf1(5))