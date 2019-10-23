#
# @lc app=leetcode id=767 lang=python3
#
# [767] Reorganize String
#
import collections, heapq
class Solution:
    def reorganizeString(self, S: str) -> str:
        if not S: return
        str_dict = collections.Counter(S)
        heap = []
        for k, v in str_dict.items():
            if v > (len(S)+1) // 2:
                return ""
            heap.append((-v,k))
        heapq.heapify(heap)
        res = []
        while len(heap) > 1:
            v1,s1 = heapq.heappop(heap)
            v2,s2 = heapq.heappop(heap)
            res += [s1,s2]
            v1 += 1
            v2 += 1
            if v1 != 0:
                heapq.heappush(heap,(v1, s1))
            if v2 != 0:
                heapq.heappush(heap,(v2, s2))
        if heap:
        
            res += [heap[0][1]]
        return "".join(res)
