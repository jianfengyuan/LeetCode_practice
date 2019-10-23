#
# @lc app=leetcode id=90 lang=python3
#
# [90] Subsets II
#
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        subsets = [[]]
        nums.sort()
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i-1]:
                l = len(subsets)
            for j in range(len(subsets) - l,len(subsets)):
                subsets.append(subsets[j]+[nums[i]])
        return subsets

