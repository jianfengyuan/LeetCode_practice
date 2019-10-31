#
# @lc app=leetcode id=713 lang=python3
#
# [713] Subarray Product Less Than K
#

# @lc code=start
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        '''
        subarray -> 连续序列,不能重新排序
        维护一个滑动窗口, left记录窗口左边界,i为窗口右边界
        如果乘积小于k,窗口左边界右移
        如果乘积大于k,窗口右边界右移
        根据窗口大小,可以计算出子数组的个数 
        [1,2,3] -> [[1,2,3],[2,3],[3]] -> i - left + 1
        '''
        if not nums: return 0 
        res = 0
        prod = 1
        left = 0
        for i in range(len(nums)):
            prod *= nums[i]
            while prod >= k and left <= i:
                prod /= nums[left]
                left += 1
            res += i - left + 1
        return res

# @lc code=end

