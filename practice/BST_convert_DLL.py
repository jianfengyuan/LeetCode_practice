# -*- coding:utf-8 -*-
"""
题目描述
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。
思路：
有序链表 => 中序遍历 ==> 左-中-右
"""
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return None
        if not pRootOfTree.left and not pRootOfTree.right:
            return pRootOfTree
        left = self.Convert(pRootOfTree.left)
        p = left
        while left and p.right:
            p = p.right
        if left:
            p.right = pRootOfTree
            pRootOfTree.left = p
        
        right = self.Convert(pRootOfTree.right)
        p = right
        while right and p.left:
            p = p.left 
        if right:
            p.left = pRootOfTree
            pRootOfTree.right = p
        
        return left if left else pRootOfTree