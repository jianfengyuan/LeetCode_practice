#
# @lc app=leetcode id=209 lang=python3
#
# [209] Minimum Size Subarray Sum
#

# @lc code=start
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        '''
        1. 建立一个比原数组长一位的sums数组，其中 sums[i]表示nums数组中[0, i - 1]的和
        2. 然后对于 sums 中每一个值 sums[i]，用二分查找法找到子数组的右边界位置，
        使该子数组之和大于 sums[i] + s
        '''
        if not nums: return 0
        temp_sum = [0]*(len(nums)+1)
        res = float('inf')
        for i in range(1,len(nums)+1):
            temp_sum[i] = temp_sum[i-1] + nums[i-1]
        
        for i in range(len(nums)):
            left = i + 1
            right = len(nums)
            key = temp_sum[i] + s
            while left <= right:
                mid = (left + right) // 2
                if temp_sum[mid] < key:
                    left = mid+1
                else:
                    right = mid - 1
            if left == len(nums)+1:
                break
            res = min(res, left - i)
        
        return res if res!= float('inf') else 0

    def minSubArrayLen2(self, s: int, nums: List[int]) -> int:
        if not nums: return 0
        l = 0
        min_len = len(nums)+1
        current_sum = 0
        for i in range(len(nums)):
            current_sum += nums[i]
            while l <= i and current_sum >= s:
                min_len = min(i-l+1, min_len)
                current_sum -= nums[l]
                l += 1
        return 0 if min_len == len(nums)+1 else min_len
# @lc code=end

