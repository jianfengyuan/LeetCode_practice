#
# @lc app=leetcode id=84 lang=python3
#
# [84] Largest Rectangle in Histogram
#

# @lc code=start
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        if not heights:return 0
        heights.append(0)
        ans = 0
        stack = [-1] # 维护递增的index数组
        for i in range(len(heights)):
            while heights[i] < heights[stack[-1]]: 
                # 当遇到比stack顶部小的高度时候,开始计算i以前的最大矩形面积
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                ans = max(ans,h*w)
            stack.append(i) 
            # 维护递增的index数组,如果高度比stack顶部还大,那添加进栈,
            # 或者直接stack中的最大已经清空,重新添加比栈顶大的值
        heights.pop()
        return ans
# @lc code=end

