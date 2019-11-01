#
# @lc app=leetcode id=81 lang=python3
#
# [81] Search in Rotated Sorted Array II
#

# @lc code=start
'''
两部分都是递增序列,且后半部分必定比前半部分小
去重: 如果中点跟左指针的值相等,那么左指针右移
'''
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if not nums: return False
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l+r)//2
            if nums[mid] == target:
                return True
            while l < mid and nums[l] == nums[mid]: ## 去重
                l += 1
            if nums[l] <= nums[mid]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r =  mid - 1
        return False

# @lc code=end

