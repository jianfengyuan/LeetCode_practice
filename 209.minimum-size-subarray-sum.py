#
# @lc app=leetcode id=209 lang=python3
#
# [209] Minimum Size Subarray Sum
#

# @lc code=start
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not s or not nums: return 0
        min_len = float("inf")
        left = 0
        current_sum = 0
        for i in range(len(nums)):
            current_sum += nums[i]
            while left < i and i - left + 1 > min_len:
                current_sum -= nums[left]
                left += 1
            if current_sum >= s:
                min_len = min(min_len, i - left + 1)
        return min_len
# @lc code=end

