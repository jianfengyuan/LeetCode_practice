#
# @lc app=leetcode id=138 lang=python3
#
# [138] Copy List with Random Pointer
#
"""
# Definition for a Node.
class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return
        dummy = Node(-1,None, None)
        current = head
        dummy_cur = dummy
        memo = {}
        while current:
            new_node = Node(current.val, None, None)
            memo[current.val] = new_node
            dummy_cur.next = new_node
            dummy_cur = dummy_cur.next
            current = current.next
        current = head
        dummy_cur = dummy.next
        while current:
            if current.random:
                dummy_cur.random = memo[current.random.val]
            dummy_cur = dummy_cur.next
            current = current.next
        return dummy.next
