#
# @lc app=leetcode id=79 lang=python3
#
# [79] Word Search
#

# @lc code=start
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board: return False
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if self.helper(board,i,j,word):
                    return True
        return False
    def helper(self,board,i,j,target):
        if len(target) == 0 : return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != target[0]:
            return False
        temp = board[i][j]
        board[i][j] = "*"      
        up = self.helper(board,i-1,j,target[1:]) 
        down = self.helper(board,i+1,j,target[1:]) 
        left = self.helper(board,i,j-1,target[1:]) 
        right = self.helper(board,i,j+1,target[1:]) 
        res = up or down or left or right
        board[i][j] = temp
        return res
# @lc code=end

