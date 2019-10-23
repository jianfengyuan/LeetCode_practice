#
# @lc app=leetcode id=80 lang=python3
#
# [80] Remove Duplicates from Sorted Array II
#
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) < 3: return len(nums)
        dummy = 1
        for i in range(1, len(nums) - 1 ):
            if nums[i - 1] != nums[i + 1]:
                nums[dummy] = nums[i]
                dummy += 1 
        nums[dummy] = nums [-1]
        return dummy +1
        
        # pre = nums[0]
        # rep = 1
        # pos = 1
        # for i in range(1, len(nums)):
        #     if nums[i] != pre or rep < 2:
        #         nums[pos] = nums[i]
        #         pos += 1
        #         if nums[i] != pre:
        #             pre = nums[i]
        #             rep = 1
        #         else:
        #             rep += 1
        # return pos

