#
# @lc app=leetcode id=416 lang=python3
#
# [416] Partition Equal Subset Sum
#
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if not nums:
            return False
        s = sum(nums)
        if s%2: return False
        target = s // 2
        dp = [[False for _ in range(target+1)] for _ in range(len(nums)+1)]
        dp[0][0] = True
        for i in range(1, len(nums)+1):
            dp[i][0] = True
        for i in range(1, len(nums)+1):
            for j in range(1, target+1):
                if j >= nums[i-1]:
                    dp[i][j] = dp[i-1][j]| dp[i-1][j-nums[i-1]]
                    ### dp[i-1][j] 不把nums[i-1]进组是否有效
                    ### dp[i-1][j-nums[i-1]] 把nums[i-1]进组是否有效
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]


