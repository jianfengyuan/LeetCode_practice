#
# @lc app=leetcode id=692 lang=python3
#
# [692] Top K Frequent Words
#
import collections,heapq
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        if not words: return
        heap = []
        word_dict = collections.Counter(words)
        for w, v in word_dict.items():
            heap.append((-v,w))
        heapq.heapify(heap)
        res = []
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
        return res
