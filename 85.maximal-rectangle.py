#
# @lc app=leetcode id=85 lang=python3
#
# [85] Maximal Rectangle
#

# @lc code=start
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        '''
        把每一行matrix看作histogram的高度,其他的按照largest rectangle in histogram的做
        '''
        
        if not matrix: return 0
        height = [0]*(len(matrix[0])+1)
        ans = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if int(matrix[i][j]):
                    height[j] += int(matrix[i][j])
                else:
                    height[j] = 0
            stack = [-1]
            for k in range(len(height)):
                while height[k] < height[stack[-1]]:
                    h = height[stack.pop()]
                    w = k - stack[-1] - 1
                    ans = max(ans,w*h)
                stack.append(k)
        return ans
# @lc code=end

