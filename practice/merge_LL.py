# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        head = ListNode(0)
        temp = head
        while pHead1 is not None and pHead2 is not None:
            if pHead1.val >= pHead2.val:
                temp.next = pHead1
                pHead1 = pHead1.next
            else:
                temp.next = pHead2
                pHead2 = pHead2.next
            temp = temp.next
        if not pHead1:
            temp.next = pHead2
        elif not pHead2:
            temp.next = pHead1
        return head.next
