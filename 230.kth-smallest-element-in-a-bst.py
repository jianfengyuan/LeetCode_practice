#
# @lc app=leetcode id=230 lang=python3
#
# [230] Kth Smallest Element in a BST
#
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        if not root: return
        self.k = k
        self.val = 0
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            if self.k > 0:
                self.k -= 1
                self.val = root.val
            dfs(root.right)
        dfs(root)
        return self.val

