#
# @lc app=leetcode id=18 lang=python3
#
# [18] 4Sum
#
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def findNnums(l, r, N, target, result, results):
            if r - l + 1 < N  or N < 2 or nums[l]*N > target \
                or nums[r] * N < target : return
            if N == 2:
                while l < r:
                    s = nums[l]+ nums[r]
                    if s == target:
                        results.append(result+[nums[l], nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l-1]:
                            l += 1
                    elif s > target:
                        r -= 1
                    else:
                        l += 1
            else:
                for i in range(l, r + 1):
                    if i == l or (i>l and nums[i] != nums[i-1] ):
                        findNnums(i + 1,r, N-1,target-nums[i],result + [nums[i]],results)


        if not nums : return
        nums.sort()
        results = []
        findNnums(0,len(nums)-1, 4, target, [], results)
        return results
