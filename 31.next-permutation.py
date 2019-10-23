#
# @lc app=leetcode id=31 lang=python3
#
# [31] Next Permutation
#
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return 
        l= r = len(nums) - 1
        while l > 0 and nums[l] <= nums[l-1]:
            l -= 1
        if l == 0:
            nums.reverse()
            return
        k = l-1
        
        i = len(nums)-1
        while nums[k] >= nums[i]:
            i-=1
        nums[k],nums[i] = nums[i], nums[k]
        while l < r:
            nums[l],nums[r] = nums[r],nums[l]
            l += 1
            r -= 1
        
        

