#
# @lc app=leetcode id=73 lang=python3
#
# [73] Set Matrix Zeroes
#

# @lc code=start
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        '''
        # brute force O(n^3)
        if not matrix: return
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    for k in range(len(matrix)):
                        if matrix[k][j] != 0:
                            matrix[k][j] = "*"
                    for k in range(len(matrix[i])):
                        if matrix[i][k] != 0:
                            matrix[i][k] = "*"
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j]=="*":
                    matrix[i][j] = 0
        return
    '''
    '''
    先对[2:n]行的元素进行遍历,如果第一行全部元素都不为0
    那只会受下面的元素影响
    如果第一行有元素为0,那无论如何第一行都为0
    先实现下面元素的影响,再回头看第一行
    O(n^2)
    '''
        if not matrix: return
        n,m,row_one_has_zeros = len(matrix), len(matrix[0]), all(matrix[0])
        for i in range(1,n):
            for j in range(m):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1,n):
            for j in range(m-1,-1,-1):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if not row_one_has_zeros:
            matrix[0] = [0]*m
        return
# @lc code=end

