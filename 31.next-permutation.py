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
        """
        如果序列是降序的，则next permutation 肯定是升序序列
        如果序列无序，则从后向前找一个出现降序的位置，确定降序位置后，从后向前找，
        找出比降序位置大的数，这两个数位置调换，然后降序位置以后的数做一次翻转
        """
        if not nums: return
        j = i = len(nums) - 1
        while i > 0 and nums[i - 1] >= nums[i]:  ## 找出前面开始递减的index
            i -= 1 
        if i == 0:  ##如果nums是倒序的，next permutation是顺序
            nums.reverse()
            return 
        k = i - 1 ## 最后 一个升序位置 nums[k] = nums[i-1] < nums[i]
        while nums[j] <= nums[k]:  ##在后面找出比nums[k]大的数
            j -= 1
        nums[j], nums[k] = nums[k], nums[j]
        l = k + 1
        r = len(nums) - 1
        while l <= r:
            nums[l],nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
        return
