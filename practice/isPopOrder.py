class Solution:
    def __init__(self):
        self.stack = []
    def IsPopOrder(self, pushV, popV):
        # write code here
        if len(pushV) != len(popV):
            return False
        pop_index = 0 
        for i in pushV:
            self.stack.append(i)
            print(self.stack,pop_index)
            while self.stack and self.stack[-1] == popV[pop_index]:
                self.stack.pop()
                print(pop_index)
                pop_index +=1
        if not self.stack:
            return True
        else:
            return False

s = Solution()
s.IsPopOrder([1,2,3,4,5],[4,5,3,2,1])