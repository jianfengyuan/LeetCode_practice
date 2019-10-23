# -*- coding:utf-8 -*-
"""
题目描述
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
解题思路：
第一次遍历：把链表中的每个节点复制一次，接在其后
第二次遍历：改变特殊指针的指向
第三次遍历：把链表拆开
"""
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if not pHead:
            return 
        current_node = pHead
        while current_node:
            # copy_node = current_node
            # current_node = current_node.next
            new_node = RandomListNode(current_node.label)
            new_node.next = current_node.next
            new_node.random = current_node.random
            current_node.next = new_node
            current_node = new_node.next
        current_node = pHead
        t =  0
        while current_node:
            if t%2:
                if current_node.random:
                    current_node.random = current_node.random.next
                current_node = current_node.next
                t += 1
                continue
            current_node = current_node.next
            t += 1
        output_head = pHead.next
        current_node = pHead
################################################################ 
        while current_node:
            copy_node = current_node.next
            current_node_next = copy_node.next
            current_node.next = current_node_next
            if current_node_next:
                copy_node.next = current_node_next.next
            else:
                copy_node.next = None
            current_node = current_node_next
        return output_head
################################################################