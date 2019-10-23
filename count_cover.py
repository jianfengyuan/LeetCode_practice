"""
题目描述
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
"""

class Solution:
    def rectCover(self, number):
        # write code here
        total = [0,1,2,3]
        while len(total) <=number:
            total.append(total[-1]+total[-2])
        return total[number]