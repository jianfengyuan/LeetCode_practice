#
# @lc app=leetcode id=106 lang=python3
#
# [106] Construct Binary Tree from Inorder and Postorder Traversal
#
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        
        '''
        preorder = [3,9,20,15,7]
        inorder = [9,3,15,20,7]
        postorder = [9,15,7,20,3]
        '''
        if not inorder or not postorder: return None
        root = TreeNode(postorder.pop())
        root_ind = inorder.index(root.val)
        root.right = self.buildTree(inorder[root_ind+1:], postorder)
        root.left = self.buildTree(inorder[:root_ind], postorder)
        
        return root
        
