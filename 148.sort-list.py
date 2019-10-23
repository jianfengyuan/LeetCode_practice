#
# @lc app=leetcode id=148 lang=python3
#
# [148] Sort List
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def merge(self,l1,l2):
        if not l1: return l2
        if not l2: return l1
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                tail = tail.next
                l1 = l1.next
            else:
                tail.next = l2
                tail = tail.next
                l2 = l2.next
        tail.next = l1 or l2
        return dummy.next

    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head
        fast = head
        slow = head
        
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        pre.next = None ##把LL从 slow处截断
        return self.merge(self.sortList(head),self.sortList(slow))



        

