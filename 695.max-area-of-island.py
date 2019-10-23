#
# @lc app=leetcode id=695 lang=python3
#
# [695] Max Area of Island
#
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        max_area = 1
        dx = [-1,0,1,0]
        dy = [0,-1,0,1]
        stack = []
        row = len(grid)
        column = len(grid[0])
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    stack.append([(i,j)])
        if not stack: return 0
        seen = set()
        while stack:
            current_path = stack.pop()
            temp_len = len(current_path)
            for j in range(len(current_path)):
                if current_path[j] in seen:
                    continue
                seen.add(current_path[j])
                current_path_x = current_path[j][0]
                current_path_y = current_path[j][1]
                for i in range(4):
                    xx = dx[i]
                    yy = dy[i]
                    n_x = current_path_x + xx
                    n_y = current_path_y + yy
                    if 0 <= n_x <= row -1 and 0 <= n_y <= column -1\
                        and  grid[n_x][n_y] == 1 and (n_x,n_y) not in current_path:
                        current_path += [(n_x,n_y)]
                        max_area = max(max_area,len(current_path))
            if temp_len != len(current_path):
                stack.append(current_path)
        return max_area


