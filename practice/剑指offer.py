class ListNode(object):
            def __init__(self, x):
                self.val = x
                self.next = None
class TreeNode:
            def __init__(self, x):
                self.val = x
                self.left = None
                self.right = None
class Solution(object):
    
    def duplicateInArray_13(self,nums):
            """
            :type nums: List[int]
            :rtype int
            """
            if not nums: return -1
            n = max(nums)+1
            tmp = [0 for _ in range(n)]
            for i in nums:
                if i < 0:
                    return -1
            for i in nums:
                if i < 0:
                    return -1
                if tmp[i] != 0:
                    return i
                tmp[i] += 1
            return -1

    def duplicateInArray_14(self,nums):
        """
        :type nums: List[int]
        :rtype int
        不修改数组，空间O(1)
        """
        def helper(nums,l,r):
            mid = (l+r)//2
            count = 0
            for i in nums:
                if i <= mid and i > l:
                    count += 1
            if count > mid-l+1:
                r = mid
            else:
                l = mid + 1
            return l,r 

        if not nums:
            return -1
        l = 1
        r = len(nums)-1
        while l!= r:
            l,r = helper(nums,l,r)
        return l

    def printLListReversingly_17(self,head):
        if not head:
            return []
        rev = None
        cur = head
        while cur.next:
            tmp = cur
            cur = cur.next
            tmp.next = rev
            rev = tmp
        output = []
        while rev:
            output.append(rev.val)
            rev = rev.next
        return output
    
    def buildTree_18(self, preorder, inorder):
            """
            :type preorder: List[int]
            :type inorder: List[int]
            :rtype: TreeNode
            """
            if not preorder or not inorder: return None
            head = preorder[0]
            if head in inorder:
                current = preorder.pop(0)
                root = TreeNode(current)
                i = inorder.index(current)
                root.left = self.buildTree_18(preorder,inorder[:i])
                root.right = self.buildTree_18(preorder,inorder[i+1:])
                return root
            return None

    def findMin_19(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return -1
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            if nums[0] > nums[1]:
                return nums[1]
            else:
                return nums[0]
        n = len(nums)-1
        while n >= 0 and nums[0]== nums[n]: ## 消除 [2,2,3,4,1,2,2,2]的情况
            n -= 1
        if nums[0] < nums[n]:
            return nums[0]
        l = 0
        r = n

        while l < r:
            mid = (l+r) // 2
            if  nums[mid] < nums[0]:
                r = mid
            else:
                l = mid + 1
        return nums[r]
            
            
    def Fibonacci_21(self,n):
            """
            :type n: int
            :rtype: int
            """
            a,b = 0,1
            for _ in range(n):
                a,b = b,a+b
            return a

    def hasPath(self, matrix, string):
        """
        :type matrix: List[List[str]]
        :type string: str
        :rtype: bool
        """
        def dfs(matrix,i,j,s):
            if matrix[i][j] != string[s]:
                return False
            if s == len(string)-1:
                return True
            memo = matrix[i][j]
            dx = [1,0,-1,0]
            dy = [0,1,0,-1]
            matrix[i][j] = "*"
            for k in range(4):
                a = i + dx[k]
                b = j + dy[k]
                if a >= 0 and a < len(matrix) and b >= 0 and b< len(matrix[a]):
                    if dfs(matrix,a,b,s+1):
                        return True
            matrix[i][j] = memo
            return False
            
        if not matrix: return False
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if dfs(matrix,i,j,0):
                    return True
        return False

        

    def vertifySequenceOfBST_46(self,sequence):
        if not sequence:return
        root = sequence[-1]
        i = 0
        for node in sequence[:-1]:
            if node > root:
                break
            i += 1
        for node in sequence[i:-1]:
            if node > root:
                return False
        left = True
        if i >0:
            left = self.vertifySequenceOfBST_46(sequence[:i])
        
        right = True
        if i < len(sequence)-2 and left:
            right = self.vertifySequenceOfBST_46(sequence[i:-1])
        return left and right

    


    def TreeConvertLL_49(self,tree):
        if not tree:
            return 
        attr = []
        def mid_order(root):
            mid_order(root.left)
            attr.append(root)
            mid_order(root.right)
        mid_order(tree)

        for i,v in enumerate(attr[:-1]):
            attr[i].right = attr[i+1]
            attr[i+1].left = v
        return attr[0]