# -*- coding:utf-8 -*-

"""
题目描述
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

若有n个台阶，则 跳法为第一下跳1阶剩余台阶的 跳法为f(n-1)，以及第一下跳2节剩余台阶跳法为f(n-2)，
以及第一下跳3阶剩余台阶 跳法为f(n-3)，依次类推知道最后一次是一下子全部跳上去，方法为1，因此f(n)=f(n-1)+f(n-2)+f(n-3)+...+f(1)+1，
f(1)=1,f(2)=2..f(n)-f(n-1)=f(n-1)因此f(n)=2f(n-1)是等比数列，首项为1，比例 为2，f(n)=2^(n-1)
"""

class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number<=0:
            return 0
        else:
            return 2**(number-1)