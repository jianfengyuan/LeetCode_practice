#
# @lc app=leetcode id=75 lang=python3
#
# [75] Sort Colors
#
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        red: 0, white: 1, blue: 2
        """
        red, white, blue = 0, 0, len(nums)-1
        while white <= blue :
            if nums[white] == 0: ##在white的index找到red，说明white的index不正确，跟red的index调换，继续寻找white的index
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1: ## 找到white的index，指针后移，这个index以后的点都应该是white
                white += 1
            else: ## 如果找到的是blue，把blue放到正确的位置，调整blue的指针，到下个循环，继续判断white的指针
                nums[blue], nums[white] = nums[white], nums[blue]
                blue -= 1
                
