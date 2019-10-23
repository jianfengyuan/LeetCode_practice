#
# @lc app=leetcode id=152 lang=python3
#
# [152] Maximum Product Subarray
#
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 0
        local_min = local_max = global_max = nums[0]
        for i in range(1,len(nums)):
            if nums[i] < 0:
                tmp = local_max
                local_max = max(local_min*nums[i],nums[i])
                local_min = min(tmp*nums[i],nums[i])
            else:
                local_max = max(local_max*nums[i],nums[i])
                local_min = min(local_min*nums[i],nums[i])
            global_max = max(global_max,local_max)
        return global_max

