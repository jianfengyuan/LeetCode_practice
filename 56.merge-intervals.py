#
# @lc app=leetcode id=56 lang=python3
#
# [56] Merge Intervals
#

# @lc code=start
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:return []
        intervals = sorted(intervals,key=lambda x:x[0])
        res = [intervals[0]]
        for i in range(1,len(intervals)):
            temp = res.pop()
            if temp[1] >= intervals[i][0]:
                new_temp = [min(temp[0], intervals[i][0])
                , max(temp[1], intervals[i][1])]
                res.append(new_temp)
                continue
            res.append(temp)
            res.append(intervals[i])
        return res
# @lc code=end

