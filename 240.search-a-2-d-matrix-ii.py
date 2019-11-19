#
# @lc app=leetcode id=240 lang=python3
#
# [240] Search a 2D Matrix II
#

# @lc code=start
'''
观察矩阵, 发现左下角和右上角,
从右上角开始,向左边递减,向下递增,
从左下角开始, 向右递增,向上递减
选一个起始点,开始遍历,结束条件为找到target
向上向右或者,向左向下查找
'''
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not target: return False
        i, j = len(matrix)-1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] == target:
                return True
            if matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
        return False
        
# @lc code=end

