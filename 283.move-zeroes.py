#
# @lc app=leetcode id=283 lang=python3
#
# [283] Move Zeroes
#

# @lc code=start
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        """
        双指针，当fast遇到0时候，fast和slow开始产生差距，slow一直跟踪0，
        可以看做0的index
        """
        if not nums: return
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[fast],nums[slow] = nums[slow], nums[fast]
                slow += 1
# @lc code=end

