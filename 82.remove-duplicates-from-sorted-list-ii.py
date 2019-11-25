#
# @lc app=leetcode id=82 lang=python3
#
# [82] Remove Duplicates from Sorted List II
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
'''
定义两个指针pre 和 current
pre每前进一步，都要用current向前遍历，
如果后面的node存在相同，pre.next != current,则pre.next = current.next
如果不存在相同,则pre.next = current 
pre继续往下走
'''
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head: return head
        dummy = ListNode(-1)
        dummy.next = head
        pre = dummy
        while pre.next:
            current = pre.next
            while current.next and current.next.val == current.val:
                current = current.next
            if pre.next != current:
                pre.next = current.next
            else:
                pre = pre.next
        return dummy.next
# @lc code=end

