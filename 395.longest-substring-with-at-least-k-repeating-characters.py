#
# @lc app=leetcode id=395 lang=python3
#
# [395] Longest Substring with At Least K Repeating Characters
#
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if not s or len(s) < k: return 0
        c = min(set(s),key=s.count)
        if s.count(c) >= k:
            return len(s)
        return max(self.longestSubstring(t,k) for t in s.split(c))
