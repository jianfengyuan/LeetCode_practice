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
其它各层的节点数目均已达最大值，且第d层所有节点从
左向右连续地紧密排列，这样的二叉树被称为完全二叉树；

完美二叉树 (Perfect Binary Tree):
二叉树的第i层至多拥有 2^(i-1)个节点数；
深度为k的二叉树至多总共有2^(k+1)-1个节点数，而总
计拥有节点数匹配的，称为“满二叉树”；

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

