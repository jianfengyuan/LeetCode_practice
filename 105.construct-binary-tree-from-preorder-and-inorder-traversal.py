#
# @lc app=leetcode id=105 lang=python3
#
# [105] Construct Binary Tree from Preorder and Inorder Traversal
#
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder or not inorder: return
        root = TreeNode(preorder.pop(0))
        root_index = inorder.index(root.val)
        root.left = self.buildTree(preorder,inorder[:root_index])
        root.right = self.buildTree(preorder, inorder[root_index+1:])
        return root

