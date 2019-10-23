#
# @lc app=leetcode id=239 lang=python3
#
# [239] Sliding Window Maximum
#
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # if not nums: return
        # res = []
        # for i in range(len(nums)-k + 1):
        #     res.append(max(nums[i:i+k]))
        # return res

        """
        维护双向队列, 双向队列存入数组下标，队列中的数字严格降序
        遍历数组，每遍历一个数字，
        1. 判断队首下标是否在窗口中，如果不在窗口中，把队首元素删掉
        2. 判断队列中是否有比该数更小的元素，如果有，把比当前数更小的元素删掉
        3. 把当前元素添加到队列尾部
        4. 当i>= k - 1时，开始记录max num in window
        """
        
        if not nums : return
        res = []
        queue = deque([])
        for i in range(len(nums)):
            if len(queue)!= 0 and i - queue[0] >= k : queue.popleft()
            while len(queue) != 0 and nums[queue[-1]] <= nums[i]: queue.pop()
            queue.append(i)
            if i >= k - 1: res.append(nums[queue[0]])
        return res
        

