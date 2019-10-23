# -*- coding:utf-8 -*-
"""
题目描述
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
中序遍历二叉搜索树可得到一个关键字的有序序列，由小到大排序。
二叉树后序遍历的特点：最后一个节点肯定是根节点。
二叉树先序遍历的特定：第一个节点肯定是根节点。
"""
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False
        root = sequence[-1] 
        # print(sequence)
        for i in range(len(sequence)):
            if sequence[i] > root:
                break
        for j in sequence[i:-1]:
            if j < root:
                return False
        
        left = True
        right = True
        if i > 0:
            left = self.VerifySquenceOfBST(sequence[:i])
        if i < len(sequence)-1:
            right = self.VerifySquenceOfBST(sequence[i:-1])
        return left and right

        

s = Solution()
l = [4,8,6,12,16,14,10]
print(s.VerifySquenceOfBST(l))