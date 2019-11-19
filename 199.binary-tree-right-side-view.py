#
# @lc app=leetcode id=199 lang=python3
#
# [199] Binary Tree Right Side View
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root: return []
        queue = [root]
        res = [root.val]
        while queue:
            for i in range(len(queue)):
                current_node = queue.pop(0)
                if current_node.left:
                    queue.append(current_node.left)
                if current_node.right:
                    queue.append(current_node.right)
            if queue: res.append(queue[-1].val)
        
        return res

    def recurse(self, root):
        if not root:return []
        self.res = [root.val]
        def helper(root):
            if not root:
                return
            if root.right:
                self.res.append(root.right.val)
                helper(root.right)
            elif root.left:
                self.res.append(root.left.val)
                helper(root.left)
        helper(root)
        return self.res
# @lc code=end

