#
# @lc app=leetcode id=5 lang=python3
#
# [5] Longest Palindromic Substring
#
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s: return ""
        length = len(s)
        start, max_len = 0, 1
        dp =  [[False]* length for _ in range(length)]
        for i in range(length):
            dp[i][i] = True
            for j in range(i):
                dp[j][i] = s[i] == s[j] and (i - j < 2 or dp[j+1][i-1]) 
                if i - j + 1 > max_len and dp[j][i]:
                    max_len = i - j + 1
                    start = j
        return s[start:start + max_len]


