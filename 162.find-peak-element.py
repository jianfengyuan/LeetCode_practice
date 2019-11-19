#
# @lc app=leetcode id=162 lang=python3
#
# [162] Find Peak Element
#

# @lc code=start
'''
1. 暴力解决
2. 二分查找
(1) 如果mid比mid+1大且比mid-1小,即mid就是peak
(2) 如果mid比mid+1大,但比mid-1小,则peak在mid 的左边,更新右边界
(3) 如果mid比mid+1小,则peak在mid的右边,更新左边界
'''
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left = 0
        right = len(nums)-1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid+1] and nums[mid] > nums[mid-1]:
                return mid
            if nums[mid] < nums[mid+1]:
                left = mid + 1
            else:
                right = mid - 1
        return left if nums[left] >= nums[right] else right
    
    def findPeakElement2(self, nums):
        if not nums: return 0
        slow = 0
        for i in range(1,len(nums)):
            if nums[i] < nums[slow]:
                return slow
            slow += 1
        return slow
# @lc code=end

