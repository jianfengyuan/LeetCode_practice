#
# @lc app=leetcode id=402 lang=python3
#
# [402] Remove K Digits
#
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        def removeleadingzero(num):
            i = 0
            while num and num[0] == "0":
                num = num[1:]
            return num if num else "0"
        if len(num) == k:
            return "0"
        
        for _ in range(k):
            to_be_removed = 0
            while to_be_removed < len(num) -1 and \
                num[to_be_removed] <= num[to_be_removed+1]:
                to_be_removed += 1
            num = num[:to_be_removed] + num[to_be_removed+1:]
        return removeleadingzero(num)


