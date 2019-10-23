# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        ans = 1
        while exponent:
            if exponent&1:
                ans *= base
            base *= base
            exponent >>= 1
        return ans

s = Solution()
print(s.Power(3.0,200))