#
# @lc app=leetcode id=92 lang=python3
#
# [92] Reverse Linked List II
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not m or not n or not head: return
        if n == m: return head
        dummyNode= ListNode(0)
        dummyNode.next = head
        pre = dummyNode
        for _ in range(m - 1):
            pre = pre.next
        
        current = pre.next
        reverse = None
        for _ in range(n-m+1):
            
            temp = current.next

            current.next = reverse
            reverse = current
            current = temp
        pre.next.next = current
        pre.next = reverse

        return dummyNode.next

