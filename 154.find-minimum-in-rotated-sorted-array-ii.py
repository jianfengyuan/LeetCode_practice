#
# @lc app=leetcode id=154 lang=python3
#
# [154] Find Minimum in Rotated Sorted Array II
#

# @lc code=start
'''
二分查找:与153同理,考虑左右两边去重
分治法: 把array对半切,如果找出递增区,并且找出递增区间之间的最小值
需要注意,当左右指针指向的值相同时候不能返回结果,eg. [3,1,3]
'''
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums: return 0
        l, r = 0, len(nums)- 1
        while l < r:
            mid = (l+r)//2
            while nums[l] == nums[mid] and mid > l:
                l += 1
            while nums[r] == nums[mid] and mid < r:
                r -= 1    
            if nums[mid] > nums[r]:
                l = mid +1
            else:
                r = mid
        return nums[r]
    
    def devide_and_conquer(self,nums,start, end):
        if end == start : return nums[start]
        if nums[start] < nums[end]: return nums[start]
        mid = (start + end)//2
        return min(self.devide_and_conquer(nums,start,mid),self.devide_and_conquer(nums,mid+1,end))
# @lc code=end

