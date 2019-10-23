# -*- coding:utf-8 -*-
"""
题目描述
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
(注意: 在返回值的list中，数组长度大的数组靠前)
"""
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        output = []
        queue = [[root]]
        while queue:
            current_path = queue.pop()
            current_node = current_path[-1]
            
            if current_node.right:
                queue.append([current_node.right])
                temp = current_path.append(current_node.right)
                if self.verify(temp,expectNumber) == 1:
                    queue.append(temp)
                elif self.verify(temp,expectNumber) == 2:
                    output.append([i.val for i in temp])
            if current_node.left:
                queue.append([current_node.left])
                temp = current_path.append(current_node.left)
                if self.verify(temp,expectNumber) == 1:
                    queue.append(temp)
                elif self.verify(temp,expectNumber) == 2:
                    output.append([i.val for i in temp])
        return sorted(output,key= lambda x:len(x),reverse=True)

    def verify(self,path,expectNumber):
        s = []
        for i in path:
            s += i.val
        if sum(s) < expectNumber:
            return 1
        elif sum(s) > expectNumber:
            return 0
        elif sum(s) == expectNumber:
            return 2