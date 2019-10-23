# -*- coding:utf-8 -*-
# 题目描述
# 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4
# 矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
# 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        length = len(matrix)
        height = len(matrix[0])
        
        # 0:➡️
        # 1:⬇️
        # 2:⬅️
        # 3:⬆️
        # #
        output = []
        while matrix:
            output+=matrix.pop(0)
            if matrix and matrix[0]:
                for row in matrix:
                    output.append(row.pop())
            if matrix:
                output+=matrix.pop()[::-1]
               
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    output.append(row.pop(0))
        return output

s = Solution()
l = [[1],[2],[3],[4],[5]]
print(s.printMatrix(l))