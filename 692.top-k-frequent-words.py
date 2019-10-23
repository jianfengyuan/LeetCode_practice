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

    def topKFrequent2(self, words: List[str], k: int) -> List[str]:
        # if not words: return
        # ## build dict
        # word_dict = {}
        # for i in range(len(words)):
        #     word_dict.setdefault(words[i], 0)
        #     word_dict[words[i]] += 1
        # ## build heap
        # heap = []
        # for word,fre in word_dict.items():
        #     if len(heap) < k:
        #         heap.append((word,fre))
        #         continue
        #     if fre > heap[0][1]:
        #         heap.append((word,fre))    
        #         self.heapify(heap, 0)
        #         heap.pop()
        # res= []
        # for _ in range(k):
        #     res.append(self.heap_pop(heap)[0])
        #     self.heapify(heap,0)
        # return res

        count = collections.Counter(words)
        heap = [(-freq, word) for word, freq in count.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]

    def heapify(self,heap,index):
        if not heap or index > len(heap) - 1:
            return
        left = index * 2 + 1
        right  = index * 2 + 2
        min_num = index
        if left < len(heap) -1 and heap[left][1] < heap[min_num][1]:
            min_num = left
        if right < len(heap) - 1 and heap[right][1] < heap[min_num][1]:
            min_num = right
        if min_num != index:
            heap[min_num], heap[index] = heap[index], heap[min_num]
            self.heapify(heap, min_num)
    def heap_pop(self,heap):
        heap[0], heap[-1] =  heap[-1], heap[0]
        out = heap.pop()
        self.heapify(heap, 0)
        return out
