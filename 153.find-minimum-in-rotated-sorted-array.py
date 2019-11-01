#
# @lc app=leetcode id=153 lang=python3
#
# [153] Find Minimum in Rotated Sorted Array
#

# @lc code=start
'''
二分法,如果mid < right, 那必定也有mid < left
如果mid > right, 那mid就在左半区
'''

class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums: return 0
        l, r = 0, len(nums)- 1
        while l < r:
            mid = (l+r)//2
            while nums[l] == nums[mid] and mid > l:
                l += 1
            if nums[mid] > nums[r]:
                l = mid +1
            else:
                r = mid
        return nums[r]
    def devide_and_conquer(self,nums,start, end):
        if nums[start] <= nums[end]: return nums[start]
        mid = (start + end)//2
        return min(self.devide_and_conquer(nums,start,mid),
        self.devide_and_conquer(nums,mid+1,end))
# @lc code=end

