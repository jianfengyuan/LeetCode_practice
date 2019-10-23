#
# @lc app=leetcode id=3 lang=python3
#
# [3] Longest Substring Without Repeating Characters
#

# @lc code=start
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        word_dict = {}
        max_len = 1
        start = 0
        for i in range(0,len(s)):
            if s[i] in word_dict:
                start,word_dict[s[i]] = max(word_dict[s[i]]+1,start),i
            else:
                word_dict.setdefault(s[i],i)
            max_len = max(max_len,i - start+1)
        return max_len


# @lc code=end

