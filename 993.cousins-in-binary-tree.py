#
# @lc app=leetcode id=993 lang=python3
#
# [993] Cousins in Binary Tree
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        '''
        可以用BFS或者DFS做,在DFS遍历树的过程中要记录target node的parent,然后做比较
        BFS的话因为是一层层遍历,所以更简单
        '''
        if not root: return False
        queue = [root]
        ans = []
        while queue:
            for _ in range(len(queue)):
                current = queue.pop(0)
                if current.left:
                    queue.append(current.left)
                    if current.left.val == x or current.left.val == y:
                        ans.append(current.val)
                if current.right:
                    queue.append(current.right)
                    if current.right.val == x or current.right.val == y:
                        ans.append(current.val)
            if len(ans) == 1:
                return False
            elif len(ans) == 2 and ans[0]!= ans[1]:
                return True
        return False
# @lc code=end

