#
# @lc app=leetcode id=25 lang=python3
#
# [25] Reverse Nodes in k-Group
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

'''
-1->1->2->3->4->5
 |        |  |
pre      cur next
reverseOneGroup():
-1->1->2->3->4->5
 |  |  |     |
start  |    end
    |  |
   last|
      cur

-1->2->1->3->4->5
 |     |  |
start  |  |
       |  |
     last |
         cur

-1->3->2->1->4->5
 |        |  |
start     | cur
          |
         last
'''

class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head and k == 1: return head
        i = 1
        dummy = ListNode(-1)
        dummy.next = head
        cur  = head
        pre = dummy
        while cur:
            if i % k == 0:
                pre = self.reverseOneGroup(pre,cur.next)
                cur = pre.next
            else:
                cur = cur.next
            i += 1
        return dummy.next
    def reverseOneGroup(self,start,end):
        last = start.next
        cur = last.next
        while cur != end:
            last.next = cur.next
            cur.next = start.next
            start.next = cur
            cur = last.next
        return last
# @lc code=end

