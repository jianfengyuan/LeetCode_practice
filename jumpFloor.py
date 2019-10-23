class Solution:
    def jumpFloor(self, number):
        """
        题目描述
        一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
        """
        # write code here
        if number == 0:
            return 1
        if number == 1:
            return 1
        return self.jumpFloor(number-1)+self.jumpFloor(number-2)

    def jumpFloor_v2(self,number):
        total = [1,1,2]
        if number <= 2:
            return total[number]
        while len(total) <= number:
            total.append(total[-1]+total[-2])
        return total[number]
s = Solution()
print(s.jumpFloor(4))