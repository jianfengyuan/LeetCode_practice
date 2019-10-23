#
# @lc app=leetcode id=95 lang=python3
#
# [95] Unique Binary Search Trees II
#
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if not n : return []
        def helper(start,end):
            trees = []
            for root in range(start,end+1):
                for left in helper(start,root-1):
                    for right in helper(root+1,end):
                        node = TreeNode(root)
                        node.left = left
                        node.right = right
                        trees.append(node)
            return trees or [None]
        return helper(1,n)

