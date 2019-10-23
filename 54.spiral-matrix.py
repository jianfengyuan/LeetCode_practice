#
# @lc app=leetcode id=54 lang=python3
#
# [54] Spiral Matrix
#
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:return
        output = []
        rows,cols = len(matrix),len(matrix[0])
        x,y = 0,0
        d = [(0,1),(1,0),(0,-1),(-1, 0)]
        i = 0
        while len(output) < rows * cols:
            if matrix[x][y] != "*":
                output.append(matrix[x][y])
                matrix[x][y] = "*"
            dx,dy = d[i%4]
            if x+dx > rows - 1 or x+dx < 0 \
                or y+dy < 0 or y+dy > cols -1\
                    or matrix[x+dx][y+dy] == "*":
                i += 1
                dx,dy = d[i%4]
            x += dx
            y += dy
        return output

