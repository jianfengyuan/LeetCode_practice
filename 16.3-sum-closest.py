#
# @lc app=leetcode id=16 lang=python3
#
# [16] 3Sum Closest
#

# @lc code=start
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if not nums:return 
        nums.sort()
        result = nums[0] + nums[1] + nums[2]
        for i in range(len(nums)-2):
            j, k = i + 1, len(nums)-1
            while  j < k:
                s = nums[i] + nums[j] + nums[k]
                if s == target:
                    return s
                if  abs(s - target) < abs(result - target):
                    result = s
                
                if s < target:
                    j += 1
                elif s > target:
                    k -= 1
        return result
# @lc code=end

