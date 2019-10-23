#
# @lc app=leetcode id=109 lang=python3
#
# [109] Convert Sorted List to Binary Search Tree
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:

        '''
        We can use the two pointer approach for finding out the middle 
        element of a linked list. Essentially, we have two pointers called 
        slow_ptr and fast_ptr. The slow_ptr moves one node at a time whereas 
        the fast_ptr moves two nodes at a time. By the time the fast_ptr 
        reaches the end of the linked list, the slow_ptr would have reached 
        the middle element of the linked list. For an even sized list, any of 
        the two middle elements can act as the root of the BST.
        '''
        if not head: return
        if not head.next: return TreeNode(head.val)
        fast = head.next.next
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        tmp = slow.next
        slow.next = None
        root = TreeNode(tmp.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(tmp.next)
        return root
