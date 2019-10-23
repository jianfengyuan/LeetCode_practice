#
# @lc app=leetcode id=74 lang=python3
#
# [74] Search a 2D Matrix
#
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]: return False
        row,col = len(matrix), len(matrix[0])
        l,r = 0,row * col - 1
        while l <= r:
            mid = (l + r) // 2
            num = matrix[mid//col][mid%col]
            if num == target:
                return True
            elif num < target:
                l = 1 + mid
            else:
                r = mid - 1
        return False
