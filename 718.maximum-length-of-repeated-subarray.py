#
# @lc app=leetcode id=718 lang=python3
#
# [718] Maximum Length of Repeated Subarray
#

# @lc code=start
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        '''
        dp 类似于最长连续子数组,substring/subarray是连续的
        因此dp中遇到A[i] != B[j]时候,直接赋0,因为subarray是连续的
        '''
        if not A or not B: return 0
        dp = [[0]*(len(A)+1) for _ in range(len(B)+1)]
        res = 0
        for i in range(1,len(dp)):
            for j in range(1,len(dp[i])):
                if B[i-1] == A[j-1]:
                    dp[i][j] = dp[i-1][j-1] +1
                else:
                    dp[i][j] = 0
                res = max(res,dp[i][j])
        return res
# @lc code=end

