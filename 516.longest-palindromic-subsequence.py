#
# @lc app=leetcode id=516 lang=python3
#
# [516] Longest Palindromic Subsequence
#

# @lc code=start
class Solution:
   
    """
    dp[i][j]为s[j:i]部分含有Palindrom的长度，关注matrix 的下半部分
    倒推 终止点为i,起始点为j
    
    """
    def longestPalindromeSubseq(self, s: str) -> int:
        if not s: return 0
        dp = [[0]*len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
            for j in range(i-1,-1,-1):
                if s[i] == s[j]:
                    dp[i][j] = dp[i-1][j+1] + 2
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j+1])
        return dp[-1][0]
# @lc code=end

