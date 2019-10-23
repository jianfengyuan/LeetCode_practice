#
# @lc app=leetcode id=24 lang=python3
#
# [24] Swap Nodes in Pairs
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head: return
        dummy = ListNode(-1)
        dummy.next = head
        slow = head
        fast = head.next
        pre = dummy
        while fast:
            slow.next = fast.next
            fast.next = slow
            pre.next = fast
            pre = fast.next
            fast = slow
            slow = slow.next
            t = 0
            while fast and t < 2:
                t += 1
                fast = fast.next
        return dummy.next
# @lc code=end

