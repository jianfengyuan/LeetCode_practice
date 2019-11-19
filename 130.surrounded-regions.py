#
# @lc app=leetcode id=130 lang=python3
#
# [130] Surrounded Regions
#

# @lc code=start
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board: return
        x = len(board[0]) - 1
        y = len(board) - 1
        for i in range(len(board)):
            if board[i][0] == "O":self.BFS(board,i,0,"$")
            if board[i][x] == "O":self.BFS(board,i,x,"$")
        for j in range(len(board[0])):
            if board[0][j] == "O":self.BFS(board,0,j,"$")
            if board[y][j] == "O":self.BFS(board,y,j,"$")
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == "O":
                    # self.BFS(board,i,j,"*")
                    board[i][j] = "X"
                if board[i][j] == "$":
                    board[i][j] = "O"
        return
       
    def BFS(self,board,i, j,target_sign):
        di = [0,1,0,-1]
        dj = [1,0,-1,0]
        board[i][j] = "$"
        for k in range(4):
            ddi = i + di[k]
            ddj = j + dj[k]
            if 0 <= ddi < len(board) and 0 <= ddj < len(board[0]) and board[ddi][ddj] == "O":
                self.BFS(board, ddi, ddj,target_sign)
# @lc code=end

