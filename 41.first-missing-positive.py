#
# @lc app=leetcode id=41 lang=python3
#
# [41] First Missing Positive
#

# @lc code=start
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        '''
        超过给定数据集大小:nums.size()的数据无需考虑
        把nums[i] 放在第nums[i]-1的位置上
        nums[nums[i]-1]==nums[i]时，即存在重复数字，已经占位，不再移动
        '''
        if not nums: return 1
        l = len(nums)
        for i in range(l):
            while 0<= nums[i] - 1 < l and nums[nums[i]-1] != nums[i]:
                tmp = nums[i]-1
                nums[i], nums[tmp] =  nums[tmp], nums[i]
        for i in range(l):
            if nums[i] != i+1:
                return i+1
        return len(nums)+1
# @lc code=end

