#
# @lc app=leetcode id=48 lang=python3
#
# [48] Rotate Image
#

# @lc code=start
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        '''
        方法1：
        当矩阵为正方形矩阵时候
        取巧方法：矩阵先沿反对角线翻转，再按水平方向翻转
        方法2：
        把矩阵复制一遍
        把i行转化未n-i-1列
        '''
        n = len(matrix)
        for i in range(n):
            for j in range(n-i):
                matrix[i][j],matrix[n-j-1][n-i-1] = matrix[n-j-1][n-i-1], matrix[i][j]
        
        for i in range(n//2):
            for j in range(n):
                matrix[i][j], matrix[-(i+1)][j] = matrix[-(i+1)][j], matrix[i][j]
        
# @lc code=end

