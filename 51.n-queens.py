#
# @lc app=leetcode id=51 lang=python3
#
# [51] N-Queens
#

# @lc code=start
'''
暴力回溯解法,因为从上往下遍历,只需要判断[0:current_queen]行是否合法就行
需要判断3个方向,正上方,左斜上方,右斜上方
'''
from copy import deepcopy
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n : return []
        
        queens = ["."*n for _ in range(n)]
        self.res = []
        
        self.helper(queens,0)
        return self.res
    
    def helper(self,queens,cur_queens):
            if cur_queens == len(queens):
                self.res.append(queens)
                return
            for i in range(len(queens)):
                if self.is_valid(queens,cur_queens,i):
                    queens[cur_queens] = queens[cur_queens][:i] + "Q" +queens[cur_queens][i+1:]
                    new_queens = deepcopy(queens)
                    self.helper(new_queens,cur_queens+1)
                    queens[cur_queens] = queens[cur_queens][:i] + "." +queens[cur_queens][i+1:]
        
    def is_valid(self,queens, row, col):
        for i in range(row):
            if queens[i][col] == "Q":
                return False
        i, j = row, col
        while i >= 0 and j >= 0:
            if queens[i][j] == "Q":
                return False
            i -= 1
            j -= 1
        i, j = row, col
        while i >= 0 and j < len(queens):
            if queens[i][j] == "Q":
                return False
            i -= 1
            j += 1
        return True
# @lc code=end

