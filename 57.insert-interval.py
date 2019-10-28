#
# @lc app=leetcode id=57 lang=python3
#
# [57] Insert Interval
#

# @lc code=start
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if not intervals or not newInterval: return [newInterval] or intervals
        left, right = [], []
        s, e = newInterval
        for i in intervals:
            if i[0] > e:
                right.append(i)
            elif i[1] < s:
                left.append(i)
            else:
                s = min(s, i[0])
                e = max(e, i[1])
        return left + [[s,e]] + right
        
    
    # def insert2(self, intervals, newInterval):
    #     if not intervals or not newInterval: return [newInterval] or intervals
    #     intervals.append(newInterval)
    #     newIntervals = sorted(intervals, key=lambda x: x[0])
    #     print(newIntervals)
    #     res = [newIntervals[0]]
    #     for i in range(1, len(newIntervals)):
    #         temp = res.pop()
    #         if newIntervals[i][0] <= temp[1]:
    #             new_temp = [min(temp[0],newIntervals[i][1]),max(newIntervals[i][1],temp[1])]
    #             res.append(new_temp)
    #             continue
    #         res.append(temp)
    #         res.append(newIntervals[i])
    #     return res
# @lc code=end

