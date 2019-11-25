#
# @lc app=leetcode id=222 lang=python3
#
# [222] Count Complete Tree Nodes
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
'''
完全二叉树(complete tree):
对于一颗二叉树，假设其深度为d（d>1）。除了第d层外，
完满二叉树 (Full Binary Tree):
所有非叶子结点的度都是2.

解法一:这道题并不需要考虑是不是完全二叉树,简单暴力求解
解法二:因为完全二叉树>完美二叉树,
因此可以求完美二叉树的深度,累计每层二叉树的节点数
加上余下的树节点
'''
class Solution:
    def countNodes2(self, root: TreeNode) -> int:
        if not root: return 0
        return 1 + self.countNodes(root.right) + self.countNodes(root.left)

    def countNodes(self,root: TreeNode) -> int:
        if not root: return 0
        leftdepth = self.get_depth(root.left)
        rightdepth = self.get_depth(root.right)
        if leftdepth == rightdepth:
            return pow(2,leftdepth) + self.countNodes2(root.right)
        else:
            return pow(2,rightdepth) + self.countNodes2(root.left)

    def get_depth(self, root):
        if not root:return 0
        return 1 + self.get_depth(root.left)
# @lc code=end

