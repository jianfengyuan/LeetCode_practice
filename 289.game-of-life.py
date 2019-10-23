#
# @lc app=leetcode id=289 lang=python3
#
# [289] Game of Life
#
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        """
        *: 1->0
        ?: 0->1
        """
        if not board: return
        padded = [[0]*(len(board[0])+2)] + \
            [[0]+board[i]+[0] for i in range(len(board))]+\
                [[0]*(len(board[-1])+2)]
        # print(padded)
        def count_windows(matrix, i, j):
            dx = [-1, -1, -1, 0, 0, 1, 1, 1]
            dy = [-1, 0, 1, -1, 1, -1, 0, 1]
            s = 0
            for ddx, ddy in zip(dx,dy):
                if matrix[i+ddx][j+ddy] == "*":
                    s += 1
                    continue
                if matrix[i+ddx][j+ddy] == "?":
                    continue
                s += matrix[i+ddx][j+ddy]
            return s
        
        for i in range(1,len(padded)-1):
            for j in range(1,len(padded[i])-1):
                windows = count_windows(padded,i,j)
                if padded[i][j] == 1:
                    if windows < 2:
                        padded[i][j] = "*"
                    elif windows > 3:
                        padded[i][j] = "*"
                elif padded[i][j] == 0:
                    if windows == 3:
                        padded[i][j] = "?"
        for i in range(1,len(padded)-1):
            for j in range(1,len(padded[i])-1):
                if padded[i][j] == "*":
                    board[i-1][j-1] = 0
                elif padded[i][j] == "?":
                    board[i-1][j-1]= 1

