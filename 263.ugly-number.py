#
# @lc app=leetcode id=263 lang=python3
#
# [263] Ugly Number
#

# @lc code=start
class Solution:
    def isUgly(self, num: int) -> bool:
        """
        不需要生成丑数数组，因为丑数都由2，3，5任意组合乘法组成，
        直接尽肯能把2，3，5的因数去掉，看最后结果是否为1
        """
        if not num: return
        while not num % 2:
            num /= 2
        while not num % 3:
            num /= 3
        while not num % 5:
            num /= 5
        return num == 1   


    def generate_ugly_nums(self,num: int)-> list:
        ugly_nums = [1]
        t2 = m2 = 0
        t3 = m3 = 0
        t5 = m5 = 0
        ugly_num = 0
        while ugly_num <= num:
            for x in range(t2, len(ugly_nums)):
                m2 = ugly_nums[x] * 2
                if m2 > ugly_nums[-1]:
                    t2 = x
                    break
            for x in range(t3, len(ugly_nums)):
                m3 = ugly_nums[x] * 3
                if m3 > ugly_nums[-1]:
                    t3 = x
                    break
            for x in range(t5, len(ugly_nums)):
                m5 = ugly_nums[x] * 5
                if m5 > ugly_nums[-1]:
                    t5 = x
                    break
            ugly_num =min(m2,m3,m5)
            ugly_nums.append(ugly_num)
        return ugly_nums
        
# @lc code=end

