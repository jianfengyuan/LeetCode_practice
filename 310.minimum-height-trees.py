#
# @lc app=leetcode id=310 lang=python3
#
# [310] Minimum Height Trees
#
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1: return [0]
        graph = [set() for _ in range(n)]
        for start, end in edges:
            graph[start].add(end)
            graph[end].add(start)
        queue = []
        leaves = [i for i in range(n) if len(graph[i]) == 1]
        while n > 2:
            n -= len(leaves)
            new_leaves = []
            for i in leaves:
                j = graph[i].pop()
                graph[j].remove(i)
                if len(graph[j]) == 1: new_leaves.append(j)
            leaves = new_leaves
        return leaves

