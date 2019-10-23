#
# @lc app=leetcode id=5 lang=python3
#
# [5] Longest Palindromic Substring
#
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        dp[i][j]表示字s[j:i]是Palindrom
        三种情况字符串未Palindrom
        1. 自己肯定是Palindrom
        2. 两个字符相邻且相同组成Palindrom
        3. 两个字符相同且dp[i-1][j+1]==True
        """
        if not s : return ""
        start, max_len = 0, 1
        dp = [[False]*len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = True
            for j in range(i):
                if s[i] == s[j] and i - j < 2:
                    dp[i][j] = True
                elif s[i] == s[j] and dp[i-1][j+1]:
                    dp[i][j] = True
                if i - j +1 > max_len and dp[i][j]:
                    max_len = i - j + 1
                    start = j
        return s[start:start+max_len]
