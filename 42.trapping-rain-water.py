#
# @lc app=leetcode id=42 lang=python3
#
# [42] Trapping Rain Water
#

# @lc code=start
class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        解法一: 一次遍历
        用两个指针指向头和尾, 找出较小的值,
        如果左边比较小,则从左向右遍历,找出比左边小的坑
        如果右边比较小,则从右向左遍历,找出比右边小的坑
        遇到比标准值大的就重新设定窗口范围
        解法二: 用stack
        如果stack为空或者当前高度小于栈顶高度,则index进栈
        如果stack里只有一个元素,即使当前高度比栈顶高度高也不能形成坑,那就从新设定边界
        如果stack里的元素大于1,则有可能形成坑,把栈顶元素弹出作为坑底,当前高度与当前栈顶
        形成坑的大小减去坑底就是坑的实际大小
        '''
        if not height: return 0
        trap = 0
        stack = []
        i, l = 0, len(height)
        while i < l:
            if not stack or height[i] <= height[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                t = stack[-1]
                stack.pop()
                if not stack: continue
                trap += (min(height[stack[-1]], height[i]) - height[t])* (i- stack[-1] - 1) 
        return trap
        
    
    def trap2(self, height: List[int]) -> int:
        if not height: return 0
        trap = 0
        left = 0
        right = len(height)-1
        while left <= right:
            m = min(height[left],height[right])
            if height[left] == m:
                left += 1
                while left < right and height[left] < m:
                    trap += (m-height[left])
                    left += 1
            if height[right] == m:
                right -= 1
                while left < right and height[right] < m:
                    trap += (m - height[right])
                    right -= 1
        return trap

# @lc code=end

