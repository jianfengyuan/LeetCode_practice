# -*- coding:utf-8 -*-
"""
题目描述
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
BFS
"""
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        output = []
        queue = [root]
        while queue:
            if queue[0].left:
                queue.append(queue[0].left)
            if queue[0].right:
                queue.append(queue[0].right)
            output.append(queue.pop(0).val)
        return output