#
# @lc app=leetcode id=8 lang=python3
#
# [8] String to Integer (atoi)
#
class Solution:
    def myAtoi(self, str: str) -> int:
        if len(str)==0:return 0
        data = str.strip()
        if not data:return 0
        temp = ["-","+","0","1","2","3","4","5","6","7","8","9"]
        if data[0] not in temp:
            return 0
        sign = 1
        if data[0] == "-":
            data = data[1:]
            sign = -1
        elif data[0] == "+":
            data = data[1:]
            sign = 1
        res = 0
        i = 0
        while i < len(data) and data[i].isdigit():
            res = res * 10 + ord(data[i]) - ord("0")
            i += 1
        return max(-2**31,min(sign*res,2**31-1))
        

