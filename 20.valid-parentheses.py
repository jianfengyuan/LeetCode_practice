#
# @lc app=leetcode id=20 lang=python3
#
# [20] Valid Parentheses
#
class Solution:
    def isValid(self, s: str) -> bool:
        if not s: return True
        stack = []
        d = {")":"(","]":"[","}":"{"}
        for p in s:
            if p in d:    
                if not stack or stack.pop() != d[p]:
                    return False
                continue
            stack.append(p)
        return len(stack) == 0

