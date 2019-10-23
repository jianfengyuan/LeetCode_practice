### 167. Two Sum II - Input array is sorted

Given an array of integers that is already **sorted in ascending order**, find two numbers such that they add up to a specific target number.
The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.

**Note:**

- Your returned answers (both index1 and index2) are not zero-based.
- You may assume that each input would have *exactly* one solution and you may not use the *same* element twice.

**Example:**

```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
```
**Answer:**

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        ## binary search
​        l = 0
​        r = len(numbers) - 1
​        while l < r:
​            if numbers[l] + numbers[r] >target:
​                r -= 1
​            elif numbers[l] + numbers[r] < target:
​                l += 1
​            else:
​                return [l+1,r+1]

```



###168. Excel Sheet Column Title

Given a positive integer, return its corresponding column title as appear in an Excel sheet.

For example:

```python
    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 
    ...
```

**Example 1:**

```
Input: 1
Output: "A"
```

**Example 2:**

```
Input: 28
Output: "AB"
```

**Answer:**

```python
class Solution:
    def convertToTitle(self, n: int) -> str:
        algeber = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        output = ""
        while n > 26:
        ## [n-1]%26
            output = algeber[(n-1)%26] + output
            n = (n-1)//26
        return algeber[(n-1)%26]+output
```



### 169. Majority Element

Given an array of size *n*, find the majority element. The majority element is the element that appears **more than** `⌊ n/2 ⌋` times.

You may assume that the array is non-empty and the majority element always exist in the array.

**Example 1:**

```
Input: [3,2,3]
Output: 3
```

**Example 2:**

```
Input: [2,2,1,1,1,2,2]
Output: 2
```

**Answer**

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        if not nums:
            return 
        appear = 1
        candidate = nums[0]
        for i in nums[1:]:
            if i == candidate:
                appear +=1 
            else:
                appear -=1
            if not appear :
                candidate = i
                appear = 1
        return candidate
```

###171. Excel Sheet Column Number

Given a column title as appear in an Excel sheet, return its corresponding column number.

For example:

```
    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    ...
```

**Example 1:**

```
Input: "A"
Output: 1
```

**Example 2:**

```
Input: "AB"
Output: 28
```

**Answer**

```python
class Solution:
    def titleToNumber(self, s: str) -> int:
        output = 0
        s = s[::-1]
        for i in range(len(s)):
            bit = ord(s[i])%65 + 1
            output += bit*(26**i)
        return output
```

### 172. Factorial Trailing Zeroes

Given an integer *n*, return the number of trailing zeroes in *n*!.

**Example 1:**

```
Input: 3
Output: 0
Explanation: 3! = 6, no trailing zero.
```

**Example 2:**

```
Input: 5
Output: 1
Explanation: 5! = 120, one trailing zero.
```

**Answer**

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        result = 0
        while n>0:
            result += n//5
            n //=5
        return result
```

### 3. Longest Substring Without Repeating Characters

Given a string, find the length of the **longest substring** without repeating characters.

**Example 1:**

```
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
```

**Example 2:**

```
Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

**Answer**

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length = 0
        substring = ""
        seen = {}
        for i,j in enumerate(s):
            if j not in substring:
                substring +=j
                max_length = max(max_length,len(substring))
                seen[j] = i
            else:
                substring = s[seen[j]:i]
                seen[j] = i
        return max_length
```



###198. House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight **without alerting the police**.

**Example 1:**

```
Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
```

**Answer**

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        self.memory = [-1]*len(nums)
        return self.rob_helper(nums,len(nums)-1)
        
    def rob_helper(self,nums,i):
        if i < 0:
            return 0
        if self.memory[i]>=0:
            return self.memory[i]
        result = max(self.rob_helper(nums,i-1),
                     self.rob_helper(nums,i-2)+nums[i])
        self.memory[i] =result
        return result
```

### 283. Move Zeroes

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Example:**

```
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        fast = 0
        slow = 0
        while fast< len(nums):
            if nums[slow] == 0:
                if nums[fast] != 0:
                    nums[slow] =nums[fast]
                    nums[fast] = 0
                    slow += 1
                else:
                    fast += 1
            else:
                slow += 1
                fast += 1
```

### 461. Hamming Distance

The [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) between two integers is the number of positions at which the corresponding bits are different.

Given two integers `x` and `y`, calculate the Hamming distance.

**Note:**
0 ≤ `x`, `y` < 231.

**Example:**

```
Input: x = 1, y = 4

Output: 2

Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

The above arrows point to positions where the corresponding bits are different.
```
**Answer:**

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        x = bin(x)[2:][::-1]
        y = bin(y)[2:][::-1]
        dis = 0
        for i in range(min(len(x),len(y))):
            dis += abs(int(x[i])-int(y[i]))
        print(dis,i)
        if len(x)>len(y):
            dis += x[i+1:].count("1")
        elif len(x)<len(y):
            dis += y[i+1:].count("1")
        return dis
        
        # ans = 0
        # while x or y:
        #   ans += (x % 2) ^ (y % 2)
        #   x /= 2
        #   y /= 2
        # return ans
```

### 617. Merge Two Binary Trees

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

**Example 1:**

```
Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
```
**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 and t2:
            return t2
        if not t2 and t1:
            return t1
        if not t1 and not t2:
            return
        t1.val = t1.val+t2.val
        t1.left = self.mergeTrees(t1.left,t2.left)
        t1.right = self.mergeTrees(t1.right,t2.right)
        return t1
```

### 206. Reverse Linked List

Reverse a singly linked list.

**Example:**

```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

**Answer**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return
        current = head.next
        new_head = ListNode(head.val)
        
        while current:
            new_node = ListNode(current.val)
            new_node.next = new_head
            new_head = new_node
            current = current.next
        return new_head
	##recursive
		    def helper(self,head:ListNode, previous=None)-> ListNode:
        if not head:
            return previous

        n = head.next
        head.next = previous
        return self.helper(n,head)
```

### 538.Convert BST to Greater Tree

Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

**Example:**

```
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
```

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.total = 0
    def convertBST(self, root: TreeNode) -> TreeNode:
###inorder walk
#         l = []
#         def inorder_walk(root):
#             if root:
#                 inorder_walk(root.left)
#                 l.append(root.val)
#                 inorder_walk(root.right)
#         def convert(root):
#             if root:
#                 root.val = sum(l[l.index(root.val):])
#                 convert(root.left)
#                 convert(root.right)
#         inorder_walk(root)
#         convert(root)
#         return root
####recursive
        if root:
            self.convertBST(root.right)
            self.total += root.val
            root.val = self.total
            self.convertBST(root.left)
        return root
```

###338. Counting Bits

Given a non negative integer number **num**. For every numbers **i** in the range **0 ≤ i ≤ num** calculate the number of 1's in their binary representation and return them as an array.

**Example 1:**

```
Input: 2
Output: [0,1,1]
```

**Example 2:**

```
Input: 5
Output: [0,1,1,2,1,2]
```

**Answer**

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        k = 0
        i = 1
        o=[0]
        while i <= num:
            if i == 2**k:
                k+=1
                o+=[1]
            else:
                o+= [o[i - 2**k]+1]
            i+=1
        return o
```

### 448. Find All Numbers Disappeared in an Array

Given an array of integers where 1 ≤ a[i] ≤ *n* (*n* = size of array), some elements appear twice and others appear once.

Find all the elements of [1, *n*] inclusive that do not appear in this array.

Could you do it without extra space and in O(*n*) runtime? You may assume the returned list does not count as extra space.

**Example:**

```
Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]
```

**Answer**

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        output = []
        for i in range(len(nums)):
            index = abs(nums[i])-1
            if nums[index] > 0:
                nums[index] = -nums[index]
        return [i+1 for i in range(len(nums)) if nums[i]>0]
```

### 543. Diameter of Binary Tree

Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the **longest** path between any two nodes in a tree. This path may or may not pass through the root.

**Example:**
Given a binary tree 

```
          1
         / \
        2   3
       / \     
      4   5    
```

Return **3**, which is the length of the path [4,2,1,3] or [5,2,1,3].

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        
        self.max_length = 1
        def helper(root):
            if root:
                left = helper(root.left)
                right = helper(root.right)
                self.max_length = max(left + right + 1,self.max_length)
                return max(left,right)+1
            else:
                return 0
        helper(root)
        return self.max_length-1
```

### 437. Path Sum III(带记忆缓存)

You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

**Example:**

```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        self.num_path = 0
        self.cache= {0:1}
        def helper(root, target,current_sum):
            
            if root :
                res = current_sum +root.val - target
                if res in self.cache:
                    self.num_path += self.cache[res]
                self.cache.setdefault(current_sum +root.val,0)
                self.cache[current_sum +root.val] += 1
                helper(root.left,target,current_sum+root.val)
                helper(root.right,target,current_sum+root.val)
                self.cache[current_sum+root.val]-=1
            return
        helper(root,sum,0)
        return self.num_path
```

### 114. Flatten Binary Tree to Linked List(困难)

Given a binary tree, flatten it to a linked list in-place.

For example, given the following tree:

```
    1
   / \
  2   5
 / \   \
3   4   6
```

The flattened tree should look like:

```
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.pre = None
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            return
        print(root.val)
        self.flatten(root.right)
        self.flatten(root.left)
        root.left = None
        root.right = self.pre
        self.pre = root
```

### 572. Subtree of Another Tree

Given two non-empty binary trees **s** and **t**, check whether tree **t** has exactly the same structure and node values with a subtree of **s**. A subtree of **s** is a tree consists of a node in **s** and all of this node's descendants. The tree **s** could also be considered as a subtree of itself.

**Example 1:**
Given tree s:

```
     3
    / \
   4   5
  / \
 1   2
```

Given tree t:

```
   4 
  / \
 1   2
```

Return true, because t has the same structure and node values with a subtree of s.



**Example 2:**
Given tree s:

```
     3
    / \
   4   5
  / \
 1   2
    /
   0
```

Given tree t:

```
   4
  / \
 1   2
```

Return false.

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
			        if not s:
            return False
        if self.match(s,t):
            return True
        return self.isSubtree(s.left,t) or self.isSubtree(s.right,t)
        
    def match(self,s,t):
        if not (s and t):
            return s is t
        return (s.val==t.val and self.match(s.left,t.left) and self.match(s.right,t.right))
```

### 238. Product of Array Except Self

Given an array `nums` of *n* integers where *n* > 1,  return an array `output` such that `output[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

**Example:**

```
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

**Note:** Please solve it **without division** and in O(*n*).

**Follow up:**
Could you solve it with constant space complexity? (The output array **does not** count as extra space for the purpose of space complexity analysis.)

**Answer**

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums: return []
        p = 1
        output= []
        for i in range(len(nums)):
            output.append(p)
            p *= nums[i]
        p = 1
        for i in range(len(nums)-1,-1,-1):
            output[i] = output[i]*p
            p*= nums[i]
            
        return output
```

### 215. Kth Largest Element in an Array

Find the **k**th largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

**Example 1:**

```
Input: [3,2,1,5,6,4] and k = 2
Output: 5
```

**Example 2:**

```
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

**Answer**

```python
class Solution:
    
    def build_min_heap(self,array):
        last_node = len(array) - 1
        parent = (last_node -1) // 2
        for i in range(parent,-1,-1):
            self.heapify(array,i)
        return array
        
    def heapify(self,heap,index):
        if index > len(heap): return
        c1 = index * 2 + 1
        c2 = index * 2 + 2
        min = index
        if c1 <= len(heap) -1 and heap[c1] < heap[min]:
            min = c1
        if c2 <= len(heap) -1 and heap[c2] < heap[min]:
            min = c2
        if min != index:
            heap[min],heap[index] = heap[index], heap[min]
            self.heapify(heap,min)
            
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = list(nums)
        heap = self.build_min_heap(nums[:k])
        for i in nums[k:]:
            if i > heap[0]:
                heap[0]=i
                self.heapify(heap,0)
        return heap[0]
```

### 347. Top K Frequent Elements

Given a non-empty array of integers, return the **k** most frequent elements.

**Example 1:**

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

**Example 2:**

```
Input: nums = [1], k = 1
Output: [1]
```

**Note:**

- You may assume *k* is always valid, 1 ≤ *k* ≤ number of unique elements.
- Your algorithm's time complexity **must be** better than O(*n* log *n*), where *n* is the array's size.

```python
class Solution:
    
    def build_heap(self,tree):
        last_node = len(tree)-1
        parent = (last_node -1)//2
        for i in range(parent,-1,-1):
            self.heapify(tree,i)
        return tree
        
    def heapify(self,tree,index):
        if index > len(tree): return
        c1 = index *2 +1
        c2 = index *2 +2
        min = index
        if c1 <= len(tree)-1 and tree[c1][1] < tree[min][1]:
            min = c1
            
        if c2 <= len(tree)-1 and tree[c2][1] < tree[min][1]:
            min = c2
        
        # print(tree,tree[c1],tree[index])
        if min != index:
            tree[min],tree[index] = tree[index],tree[min]
            self.heapify(tree,min)
    
    def delete_from_heap(self,heap,index):
        if index>len(heap)-1: return
        last_node = len(heap)-1
        heap[index],heap[last_node] = heap[last_node],heap[index]
        d = heap.pop()
        self.heapify(heap,index)
        return d
    
    
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return
        num_dic = {}
        heap = []
        output = []
        for i in nums:
            num_dic.setdefault(i,0)
            num_dic[i] += 1
        print(num_dic)
        heap = self.build_heap(list(num_dic.items())[:k])
        for i,j in list(num_dic.items())[k:]:
            if j > heap[0][1]:
                heap[0] = (i,j)
                self.heapify(heap,0)
        t = 0
        
        while t<k:
            output = output+[self.delete_from_heap(heap,0)[0]]
            t+=1
        return output
```

### 102. Binary Tree Level Order Traversal

Given a binary tree, return the *level order* traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```



return its level order traversal as:

```
[
  [3],
  [9,20],
  [15,7]
]
```
**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        queue = [root]
        output = []
        current_level = []
        while queue:
            current_level = []
            for _ in range(len(queue)):
                current = queue.pop(0)
                current_level.append(current.val)
                if current.left:
                    queue.append(current.left)
                if current.right:
                    queue.append(current.right)
            output+=[current_level]
        return output
```



### 581. Shortest Unsorted Continuous Subarray

Given an integer array, you need to find one **continuous subarray** that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order, too.

You need to find the **shortest** such subarray and output its length.

**Example 1:**

```
Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
```

**Answer**

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        temp = [a==b for a,b in zip(nums,sorted(nums))]
        return 0 if all(temp) else len(nums) - temp.index(False) - temp[::-1].index(False)
```

### 55. Jump Game(GREEDY)

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

**Example 1:**

```
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**

```
Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.
```

**Answer**

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        farest = 0
        for i in range(len(nums)-1):
            if i > farest:
                return False
            farest = max(farest,i+nums[i])
        return farest>=len(nums)-1
```



### 98. Validate Binary Search Tree

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

**Example 1:**

```
Input:
    2
   / \
  1   3
Output: true
```

**Example 2:**

```
    5
   / \
  1   4
     / \
    3   6
Output: false
Explanation: The input is: [5,1,4,null,null,3,6]. The root node's value
             is 5 but its right child's value is 4.
```

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        pass
        
    def isValidBST(self, root: TreeNode,left=float("inf"),right=float("-inf")) -> bool:
        if not root:return True
        if root.val >= left or root.val <= right:
            return False
        return self.isValidBST(root.left,min(left,root.val),right) and self.isValidBST(root.right,left,max(root.val,right))
```

###739. Daily Temperatures

Given a list of daily temperatures `T`, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put `0` instead.

For example, given the list of temperatures `T = [73, 74, 75, 71, 69, 72, 76, 73]`, your output should be `[1, 1, 4, 2, 1, 1, 0, 0]`.

**Note:** The length of `temperatures` will be in the range `[1, 30000]`. Each temperature will be an integer in the range `[30, 100]`.

**Answer**

```python
from collections import defaultdict
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = []
        out = [0]*len(T)
        for i in range(len(T)-1,-1,-1):
            while stack and T[i] >= T[stack[-1]]:
                stack.pop()
            if stack:
                out[i] = stack[-1]-i
            stack.append(i)
        return out
```

### 230. Kth Smallest Element in a BST

Given a binary search tree, write a function `kthSmallest` to find the **k**th smallest element in it.

**Note:** 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

**Example 1:**

```
Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
```

**Example 2:**

```
Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3
```

**Follow up:**
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # l = []
        # def preorder(root):
        #     if not root:
        #         return
        #     if root.left:
        #         preorder(root.left)
        #     l.append(root.val)
        #     if root.right:
        #         preorder(root.right)
        # preorder(root)
        # return(l[k-1])
        self.k = k
        self.res = 0
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            if self.k >0:
                self.k -= 1
                self.res = root.val
            dfs(root.right)
        dfs(root)
        return self.res
```

### 378. Kth Smallest Element in a Sorted Matrix(难)

Given a *n* x *n* matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

**Example:**

```
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
```

**Note:** 
You may assume k is always valid, 1 ≤ k ≤ n2.

**Answer**

https://www.cnblogs.com/grandyang/p/5727892.html

```python
import math
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        ########heap sort(RTE)################
        # t = 0
        # heap = []
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[i])):
        #         if t < k:
        #             t +=1
        #             self.insert(heap,matrix[i][j])
        #             continue
        #         if matrix[i][j] <heap[0]:
        #             heap[0] = matrix[i][j]
        #             self.heapify(0,heap)
        # return heap[0]
        #####################################
        ##########binary search##############
        if k == 1:return matrix[0][0]
        if k == len(matrix)*len(matrix[0]): return matrix[-1][-1]
        lo,hi = matrix[0][0],matrix[-1][-1]
        while lo < hi:
            mid = lo+(hi-lo)//2
            count = self.get_num_equal(mid,matrix)
            if count < k:
                lo = mid+1
            else:
                hi = mid
        return lo
            
    def get_num_equal(self,num,matrix):
        res = 0
        i = len(matrix)-1
        j = 0
        while j<len(matrix[0]) and i >=0:
            if matrix[i][j] <= num:
                res += i+1
                j +=1
            else:
                i -= 1
        return res
    
    def heapify(self,index,tree):
        if index > len(tree):
            return
        c1 = index*2 + 1
        c2 = index*2 +2
        max = index
        if c1 <= len(tree)-1 and tree[c1]>tree[max]:
            max = c1
        if c2 <= len(tree)-1 and tree[c2]>tree[max]:
            max = c2
        if max != index:
            index, max = max,index
            self.heapify(max,tree)
    
    def build_heap(self,tree):
        pass
    def insert(self,heap,num):
        heap.append(num)
        if len(heap)>1:
            self.heap_up(heap,num,len(heap)-1)
    def heap_up(self,tree,num,index):
        if index > 0:
            parent = math.floor((index-1)/2)
            if tree[parent]<num:
                tree[parent],tree[index] = tree[index],tree[parent]
                self.heap_up(tree,num,parent)
```



### 15. 3Sum (经典)

Given an array `nums` of *n* integers, are there elements *a*, *b*, *c* in `nums` such that *a* + *b* + *c* = 0? Find all unique triplets in the array which gives the sum of zero.

**Note:**

The solution set must not contain duplicate triplets.

**Example:**

```
Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

**思路：**

先把数组排序，遍历数组,a = nums[i],那么从i+1: -1之间寻找b，c，使用两个指针寻找b+c = -a的组合，转化成2-sum

去重：因为数组是升序，因此在找到某一组合，左右指针移动时，判断这个数与上一个数是否相等，如果相等则继续左移或右移指针

**Answer**

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        answer = []
        for i in range(len(nums)):
            a = nums[i]
            if i!=0 and nums[i-1] == a:
                continue
            left = i+1
            right = len(nums)-1
            while left < right:
                b = nums[left]
                c = nums[right]
                total = a + b + c
                if total ==0:
                    # left+=1
                    # right-=1
                    answer.append([a,b,c])
                    while left<right and nums[left]==nums[left+1]:
                        left+=1
                    while right>left and nums[right]==nums[right-1]:
                        right -= 1
                    left+=1
                    right-=1
                    
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        return answer
```

### 113. Path Sum II

Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

**Note:** A leaf is a node with no children.

**Example:**

Given the below binary tree and `sum = 22`,

```
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \    / \
7    2  5   1
```

Return:

```
[
   [5,4,11,2],
   [5,8,4,5]
]
```

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
############### BFS  #####################
        if not root: return []
        answer = []
        queue = [root]
        path =[[root.val]]
        while queue:
            current = queue.pop(0)
            current_path = path.pop(0)
            if current.left:
                queue.append(current.left)
                path.append(current_path+[current.left.val])
            if current.right:
                queue.append(current.right)
                path.append(current_path+[current.right.val])
            if not current.left and not current.right and sum(current_path)==target:
                answer.append(current_path)
################ recursive ####################
	        def helper(root,path,target):
            if not root:
                return
            if sum(path)+root.val != target or root.right or root.left:
                helper(root.left,path+[root.val],target)
                helper(root.right,path+[root.val],target)
            if not root.right and not root.left and sum(path)+root.val==target:
                answer.append(path+[root.val])
        helper(root,[],target)
        return answer
        return answer
```

### 11. Container With Most Water

Given *n* non-negative integers *a1*, *a2*, ..., *an* , where each represents a point at coordinate (*i*, *ai*). *n* vertical lines are drawn such that the two endpoints of line *i* is at (*i*, *ai*) and (*i*, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

**Note:** You may not slant the container and *n* is at least 2.

![img](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

**Answer**

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        left = 0
        right = len(height)-1
        while left < right:
            max_area = max(max_area,(right-left)*min(height[left],height[right]))
            if height[left]<=height[right]:
                left +=1
            else:
                right -= 1
        return max_area
```

### 19. Remove Nth Node From End of List

Given a linked list, remove the *n*-th node from the end of list and return its head.

**Example:**

```
Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
```

**Note:**

Given *n* will always be valid.

**思路**

快慢指针

**Answer**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        slow = head
        fast = head
        count = 0
        while fast.next:
            fast = fast.next
            count += 1
            if count > n:
                slow = slow.next
        if count >= n:
            slow.next = slow.next.next
        else:
            head = slow.next
        return head
```



### 33. Search in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`).

You are given a target value to search. If found in the array return its index, otherwise return `-1`.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Example 2:**

```
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

**思路**

二分法，因为数组是从递增序列旋转而来，因此有以下特点，设三个点，lo，hi，mid

当 lo < mid, 

如果nums[lo] < target < nums[mid]  => target在lo和mid之间，lo和mid之间是一个递增区间，因此hi = mid - 1

在当前条件下，另外两种情况是target < nums[lo] 和target > nums[mid]，

target < nums[lo] => 因为lo是左半部分最小，因此target比左半部分更小，只能从mid的右半部分找，因此lo = mid +1

target > nums[mid] => 因为lo < mid，因此要从mid的右边找左半部分的更大值，因此lo= mid + 1 

当lo > mid

此时lo和mid之间是两个递增数组，无法判断数之间的关系，但是可以确定，mid和hi之间是递增数组

如果nums[mid] < target <= nums[hi] => target在mid和hi的递增区间之间，因此，lo = mid + 1

在当前条件下，另外两种情况是target < nums[mid] 和 target > nums[hi]:

target > nums[hi] => hi是右半部分最大，因此target比右半部分更大应从mid的左半部分找，因此，hi = mid - 1

target < nums[mid] =>target比mid小，只能从mid的左边找，因此hi = mid - 1

**Answer**

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        h = len(nums) -1
        while l <= h:
            mid = (l+h)//2
            if nums[mid] == target:
                return mid
            
            if nums[l] <= nums[mid]:
                if nums[l]<=target<=nums[mid]:
                    h = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] <= target <= nums[h]:
                    l = mid + 1
                else:
                    h = mid - 1
        return -1
```

### 34. Find First and Last Position of Element in Sorted Array

Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

If the target is not found in the array, return `[-1, -1]`.

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Example 2:**

```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

**Answer**

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums: return [-1,-1]
        i = 0
        j = len(nums)-1
        output = [-1,-1]
        while i < j:
            mid = (i+j) //2
            if nums[mid] < target:
                i = mid +1
            else:
                j = mid
        if nums[i]!= target:
            return output
        else:
            output[0] = i
        j = len(nums)-1
        while i < j:
            mid = (i + j) //2+1
#####################################################################
#Why does this trick work? When we use mid = (i+j)/2, the mid is rounded to the lowest integer. In other words, mid is always biased towards the left. This means we could have i == mid when j - i == mid, but we NEVER have j == mid. So in order to keep the search range moving, you must make sure the new i is set to something different than mid, otherwise we are at the risk that i gets stuck. But for the new j, it is okay if we set it to mid, since it was not equal to mid anyways. Our two rules in search of the left boundary happen to satisfy these requirements, so it works perfectly in that situation. Similarly, when we search for the right boundary, we must make sure i won't get stuck when we set the new i to i = mid. The easiest way to achieve this is by making mid biased to the right, #
#####################################################################
            if nums[mid] > target:
                j = mid - 1
            else:
                i = mid 
        output[1] = j
        return output
```



### 297. Serialize and Deserialize Binary Tree

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Example:** 

```
You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
```

**Clarification:** The above format is the same as [how LeetCode serializes a binary tree](https://leetcode.com/faq/#binary-tree). You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

**Note:** Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.

**Answer**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        data = []
        def helper(root):
            if not root:
                data.append("#")
                return
            data.append(str(root.val))
            helper(root.left)
            helper(root.right)
        helper(root)
        return " ".join(data)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        
        def helper():
            val = next(data)
            if val == "#":
                return
            node = TreeNode(int(val))
            node.left = helper()
            node.right = helper()
            return node
        data = iter(data.split())
        return helper()
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```



### 105. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.

**Note:**
You may assume that duplicates do not exist in the tree.

For example, given

```python
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
```

Return the following binary tree:

```
    3
   / \
  9  20
    /  \
   15   7
```

**Answer:**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder or not inorder:return      
        root_num = preorder.pop(0)
        root = TreeNode(root_num)
        root_index = inorder.index(root_num)
        left = self.buildTree(preorder,inorder[:root_index])
        right = self.buildTree(preorder,inorder[root_index+1:])
        root.left = left
        root.right = right
        return root
```





## 动态规划：

### 746. Min Cost Climbing Stairs

On a staircase, the `i`-th step has some non-negative cost `cost[i]` assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

**Example 1:**

```
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
```



**Example 2:**

```
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
```



**<u>状态方程</u>**：f[i] is the final cost to climb to the top from step i. Then f[i] = cost[i] + min(f[i+1], f[i+2]).
**Answer**
```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0]*len(cost)
        dp[0]=cost[0]
        dp[1]=cost[1]
        for i in range(2,len(cost)):
            dp[i] = min(dp[i-1],dp[i-2])+cost[i]
        return min(dp[-1],dp[-2])
```

### 877. Stone Game

Alex and Lee play a game with piles of stones.  There are an even number of piles **arranged in a row**, and each pile has a positive integer number of stones `piles[i]`.

The objective of the game is to end with the most stones.  The total number of stones is odd, so there are no ties.

Alex and Lee take turns, with Alex starting first.  Each turn, a player takes the entire pile of stones from either the beginning or the end of the row.  This continues until there are no more piles left, at which point the person with the most stones wins.

Assuming Alex and Lee play optimally, return `True` if and only if Alex wins the game.

 

**Example 1:**

```
Input: [5,3,4,5]
Output: true
Explanation: 
Alex starts first, and can only take the first 5 or the last 5.
Say he takes the first 5, so that the row becomes [3, 4, 5].
If Lee takes 3, then the board is [4, 5], and Alex takes 5 to win with 10 points.
If Lee takes the last 5, then the board is [3, 4], and Alex takes 4 to win with 9 points.
This demonstrated that taking the first 5 was a winning move for Alex, so we return true.
```

 

**Note:**

1. `2 <= piles.length <= 500`
2. `piles.length` is even.
3. `1 <= piles[i] <= 500`
4. `sum(piles)` is odd.



***状态方程***：

```python
f(i,j) = max(reword[i]+min(f(i+2,j),f(i+1,j-1)),
             reword[j]+min(f(i,j-2),f(i+1,j-1)))
```
**Answer**

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        win = sum(piles)/2
        cache = {}
        def help(i,j):
            if i>=j:
                return 0
            if j==i+1 and j <len(piles):
                return piles[i]
            if (i,j) in cache.keys():
                return cache[(i,j)]
            else:
                max_num = max(piles[i]+min(help(i+2,j),help(i+1,j-1)),
                              piles[j-1]+min(help(i,j-2),help(i+1,j-1)))
                cache[(i,j)] = max_num
            return max_num
        # print(help(0,len(piles)))
        return help(0,len(piles)) > win
```



### 931. Minimum Falling Path Sum

Given a **square** array of integers `A`, we want the **minimum** sum of a *falling path*through `A`.

A falling path starts at any element in the first row, and chooses one element from each row.  The next row's choice must be in a column that is different from the previous row's column by at most one.

 

**Example 1:**

```
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: 12
Explanation: 
The possible falling paths are:
```

- `[1,4,7], [1,4,8], [1,5,7], [1,5,8], [1,5,9]`
- `[2,4,7], [2,4,8], [2,5,7], [2,5,8], [2,5,9], [2,6,8], [2,6,9]`
- `[3,5,7], [3,5,8], [3,5,9], [3,6,8], [3,6,9]`

The falling path with the smallest sum is `[1,4,7]`, so the answer is `12`.



***状态方程***：

```python
dp[i][j] = min(dp[i-1][j],dp[i-1][j-1],dp[i-1][j+1])+A[i][j]
```
**Answer**
```python
class Solution:
    def minFallingPathSum(self, A: List[List[int]]) -> int:
        ##
        min_num = 0
        dp = [A[0]]+[[0]*(len(A[0])) for _ in range(len(A)-1)]
        for i in range(1,len(dp)):
            for j in range(0,len(dp[0])):
                if j !=0 and j+1 < len(A):
                    dp[i][j] = min(dp[i-1][j],dp[i-1][j-1],dp[i-1][j+1])+A[i][j]
                elif j==0:
                    dp[i][j] = min(dp[i-1][j],dp[i-1][j+1])+A[i][j]
                elif j+1==len(A):
                    dp[i][j] = min(dp[i-1][j],dp[i-1][j-1])+A[i][j]
                
        return min(dp[-1])
```

### 983. Minimum Cost For Tickets

In a country popular for train travel, you have planned some train travelling one year in advance.  The days of the year that you will travel is given as an array `days`.  Each day is an integer from `1` to `365`.

Train tickets are sold in 3 different ways:

- a 1-day pass is sold for `costs[0]` dollars;
- a 7-day pass is sold for `costs[1]` dollars;
- a 30-day pass is sold for `costs[2]` dollars.

The passes allow that many days of consecutive travel.  For example, if we get a 7-day pass on day 2, then we can travel for 7 days: day 2, 3, 4, 5, 6, 7, and 8.

Return the minimum number of dollars you need to travel every day in the given list of `days`.

 

**Example 1:**

```
Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
Explanation: 
For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 1-day pass for costs[0] = $2, which covered day 1.
On day 3, you bought a 7-day pass for costs[1] = $7, which covered days 3, 4, ..., 9.
On day 20, you bought a 1-day pass for costs[0] = $2, which covered day 20.
In total you spent $11 and covered all the days of your travel.
```

**Example 2:**

```
Input: days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
Output: 17
Explanation: 
For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 30-day pass for costs[2] = $15 which covered days 1, 2, ..., 30.
On day 31, you bought a 1-day pass for costs[0] = $2 which covered day 31.
In total you spent $17 and covered all the days of your travel.
```

**状态方程**

```py
dp[i]=min(dp[max(0,i-7)]+costs[1],dp[max(0,i-1)]+costs[0],dp[max(0,i-30)]+costs[2])
```

**Answer**

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        dp=[0 for i in range(days[-1]+1)]
        for i in range(days[-1]+1):
             if i not in days:
                dp[i]=dp[i-1]
             else:
                dp[i]=min(dp[max(0,i-7)]+costs[1],dp[max(0,i-1)]+costs[0],dp[max(0,i-30)]+costs[2])
        return dp[-1]
```

### 712. Minimum ASCII Delete Sum for Two Strings

Given two strings `s1, s2`, find the lowest ASCII sum of deleted characters to make two strings equal.

**Example 1:**

```
Input: s1 = "sea", s2 = "eat"
Output: 231
Explanation: Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.
Deleting "t" from "eat" adds 116 to the sum.
At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.
```



**Example 2:**

```
Input: s1 = "delete", s2 = "leet"
Output: 403
Explanation: Deleting "dee" from "delete" to turn the string into "let",
adds 100[d]+101[e]+101[e] to the sum.  Deleting "e" from "leet" adds 101[e] to the sum.
At the end, both strings are equal to "let", and the answer is 100+101+101+101 = 403.
If instead we turned both strings into "lee" or "eet", we would get answers of 433 or 417, which are higher.
```

**状态方程**

```python
dp[i][j] = min(dp[i+1][j]+ ord(s1[i]), dp[i][j+1] + ord(s2[j]))
```

**Answer**

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
        for i in range(len(s1)-1,-1,-1):
            dp[i][len(s2)] = dp[i+1][len(s2)] + ord(s1[i])
        for j in range(len(s2)-1,-1,-1):
            dp[len(s1)][j] = dp[len(s1)][j+1] + ord(s2[j])
        for i in range(len(s1)-1,-1,-1):
            for j in range(len(s2)-1,-1,-1):
                if s1[i] == s2[j]:
                    dp[i][j] = dp[i+1][j+1]
                else:
                    dp[i][j] = min(dp[i+1][j]+ ord(s1[i]),
                                  dp[i][j+1] + ord(s2[j]))
        return dp[0][0]
```

### 714. Best Time to Buy and Sell Stock with Transaction Fee

Your are given an array of integers `prices`, for which the `i`-th element is the price of a given stock on day `i`; and a non-negative integer `fee` representing a transaction fee.

You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction. You may not buy more than 1 share of a stock at a time (ie. you must sell the stock share before you buy again.)

Return the maximum profit you can make.

**Example 1:**

```
Input: prices = [1, 3, 2, 8, 4, 9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
Buying at prices[0] = 1Selling at prices[3] = 8Buying at prices[4] = 4Selling at prices[5] = 9The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
```



**Note:**

`0 < prices.length <= 50000`.

`0 < prices[i] < 50000`.

`0 <= fee < 50000`.

**题目解释**

我们用现金来记录我们赚到的利润，同时我们需要记录手中的股票价格，然后每次需要计算当前的现金和如果出售股票所带来了的利润，如果这个利润比当前的现金多的话，就售出。同时每次也需要手中的股票价格和再次购买股票之后剩下的钱，如果购买完股票之后手中的钱还没有当前的股票价值多，就不需要变化了。

举个例子：[1,3,7,5,10,3]，手续费为3。

刚开始假设我们手中的现金cash为0，并且手中持股hold为-1，这个-1就是第0天的价格，我们假设购买了这个股票，因为最初在第0天我们手中并没有现金，所以这个股票的价值就是负的，就像借别人的钱买的一样，在刚开始是负的。

然后从第1天开始，当前的股票价格为3，如果出售的话此时利润为：3+-1-3=-1，利润比手中的现金还少，所以不需要变化，此时cash=0，hold=-1。

第2天，当前的股票价格为7，出售的话利润为：7-1-3=3，利润比手中的现金多，所以就进行出售，此时cash=3，但是如果出售完成之后我们购买第2天的股票，此时如果用手中的现金去买的话，还会剩下3-7=-4，购买完之后剩下的现金还不如当前的持有的股票（-1）值钱，所以就不够买，手中的股票还是-1。那么这个cash就是假设出售的价格，以后或许会发生变化。此时cash=3，hold=-1。

第3天，股票价格为5，出售的话利润为：5-1-3=1，利润比手中的现金少，所以cash保持不变，而且3-5=-2，小于手中股票的价值，所以cash=3，hold=-1。

第4天，股票价格为10，出售的话利润为：10-1-3=6，利润比手中的现金多，所以cash更新为6，hold依旧不变。

第5天，同理保持不变，所以最终得到的利润为6。

在这个过程中，手中现金和持有股票都会不断的变化，手中的现金是在假设此时出售股票时所带来的利润。如果以后的利润会更大，那么就更新这个数值，最终求出一个最大利润。
**Answer**

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        cash = 0
        hold = -prices[0]
        for price in prices[1:]:
            cash = max(cash, hold+price-fee)
            hold = max(hold,cash-price)
            print(price,cash,hold)
        return cash
```

### 646. Maximum Length of Pair Chain

You are given `n` pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair `(c, d)` can follow another pair `(a, b)` if and only if `b < c`. Chain of pairs can be formed in this fashion.

Given a set of pairs, find the length longest chain which can be formed. You needn't use up all the given pairs. You can select pairs in any order.

**Example 1:**

```
Input: [[1,2], [2,3], [3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4]
```



**Note:**

1. The number of given pairs will be in the range [1, 1000].

**解释：**

先把pairs按第一位排序，确定pairs之间的联系，然后对第二位进行dp，最长pair chain 是该pair前的能与该pair组成pair chain的最长pair chain长度加1，用一个长度为n的dp table记录最长长度

**状态方程：**

```python
dp[i] = max(dp[i],dp[j]+1) (j<i)
```



**Answer**

```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs = sorted(pairs,key=lambda x:x[0])
        dp = [1 for _ in range(len(pairs))]
        for i in range(len(pairs)):
            for j in range(i):
                if pairs[j][1] < pairs[i][0]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)
```



### 494. Target Sum

You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+`and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

**Example 1:**

```
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
```



**Note:**

1. The length of the given array is positive and will not exceed 20.
2. The sum of elements in the given array will not exceed 1000.
3. Your output answer is guaranteed to be fitted in a 32-bit integer.

**状态方程**

![image-20190422162011905](/Users/jianfengyuan/Library/Application Support/typora-user-images/image-20190422162011905.png)

```python
dp[i+1][j+nums[i]] += dp[i][j]
dp[i+1][j-nums[i]] += dp[i][j]
```

**Answer**

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        s = sum(nums)
        if s < S: return 0
        min_s = -s
        offset = s
        dp = [[0 for _ in range(min_s,s+1)] for _ in range(len(nums)+1)]
        dp[0][s] = 1
        for i in range(len(nums)):
            for j in range(len(dp[0])):
                if dp[i][j]:
                    dp[i+1][j+nums[i]] += dp[i][j]
                    dp[i+1][j-nums[i]] += dp[i][j]
        return dp[-1][S+offset]
```

### 638. Shopping Offers

In LeetCode Store, there are some kinds of items to sell. Each item has a price.

However, there are some special offers, and a special offer consists of one or more different kinds of items with a sale price.

You are given the each item's price, a set of special offers, and the number we need to buy for each item. The job is to output the lowest price you have to pay for **exactly** certain items as given, where you could make optimal use of the special offers.

Each special offer is represented in the form of an array, the last number represents the price you need to pay for this special offer, other numbers represents how many specific items you could get if you buy this offer.

You could use any of special offers as many times as you want.

**Example 1:**

```
Input: [2,5], [[3,0,5],[1,2,10]], [3,2]
Output: 14
Explanation: 
There are two kinds of items, A and B. Their prices are $2 and $5 respectively. 
In special offer 1, you can pay $5 for 3A and 0B
In special offer 2, you can pay $10 for 1A and 2B. 
You need to buy 3A and 2B, so you may pay $10 for 1A and 2B (special offer #2), and $4 for 2A.
```



**Example 2:**

```
Input: [2,3,4], [[1,1,0,4],[2,2,1,9]], [1,2,1]
Output: 11
Explanation: 
The price of A is $2, and $3 for B, $4 for C. 
You may pay $4 for 1A and 1B, and $9 for 2A ,2B and 1C. 
You need to buy 1A ,2B and 1C, so you may pay $4 for 1A and 1B (special offer #1), and $3 for 1B, $4 for 1C. 
You cannot add more items, though only $9 for 2A ,2B and 1C.
```



**Note:**

1. There are at most 6 kinds of items, 100 special offers.
2. For each item, you need to buy at most 6 of them.
3. You are **not** allowed to buy more items than you want, even if that would lower the overall price

**状态方程**

```python
res = min(self.shoppingOffers(price,special,needs)+offer[-1],res)
```

先计算不使用special offer一共需要花多少钱，然后在递归寻找最划算的special offer

**Answer**

```python
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        res = 0
        for i in range(len(price)):
            res += needs[i]*price[i]
        for offer in special:
            is_valid = True
            for j in range(len(needs)):
                if needs[j] < offer[j]:
                    is_valid = False
                needs[j] -= offer[j]
            if is_valid:
                res = min(self.shoppingOffers(price,special,needs)+offer[-1],res)
            for j in range(len(needs)):
                needs[j] += offer[j]
        return res
```



### 486. Predict the Winner

Given an array of scores that are non-negative integers. Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on. Each time a player picks a number, that number will not be available for the next player. This continues until all the scores have been chosen. The player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner. You can assume each player plays to maximize his score.

**Example 1:**

```
Input: [1, 5, 2]
Output: False
Explanation: Initially, player 1 can choose between 1 and 2. 
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 chooses 5, then player 1 will be left with 1 (or 2). 
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5. 
Hence, player 1 will never be the winner and you need to return False.
```



**Example 2:**

```
Input: [1, 5, 233, 7]
Output: True
Explanation: Player 1 first chooses 1. Then player 2 have to choose between 5 and 7. No matter which number player 2 choose, player 1 can choose 233.
Finally, player 1 has more score (234) than player 2 (12), so you need to return True representing player1 can win.
```



**Note:**

1. 1 <= length of the array <= 20.
2. Any scores in the given array are non-negative integers and will not exceed 10,000,000.
3. If the scores of both players are equal, then player 1 is still the winner.

**状态方程**

```python
res = max(nums[i]+min(helper(nums,i+1,j-1),helper(nums,i+2,j)),
          nums[j]+min(helper(nums,i+1,j-1),helper(nums,i,j-2)))
```



**Answer**

```python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        memory = {}
        def helper(nums,i,j):
            if i > j :
                return 0
            if (i,j) in memory:
                return memory[(i,j)]
            res = max(nums[i]+min(helper(nums,i+1,j-1),helper(nums,i+2,j)),
                     nums[j]+min(helper(nums,i+1,j-1),helper(nums,i,j-2)))
            memory[(i,j)] = res
            return res
        p1 = helper(nums,0,len(nums)-1)
        p2 = sum(nums) - p1
        return p1>=p2
```

### 279. Perfect Squares

Given a positive integer *n*, find the least number of perfect square numbers (for example, `1, 4, 9, 16, ...`) which sum to *n*.

**Example 1:**

```
Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
```

**Example 2:**

```
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```

**状态方程**

每个数最长的perfect square和为1的倍数，因此可以不断优化每个数对应的组成数

```python
dp[i]=min(dp[i],dp[i-sq]+1)
```
**Answer**

```python
import math
class Solution:
    def numSquares(self, n: int) -> int:
        perfect_square = [i*i for i in range(1,int(math.sqrt(n))+1)]
        dp = [i for i in range(n+1)]
        for i in range(1,n+1):
            for sq in perfect_square:
                if i - sq < 0:
                    break
                dp[i]=min(dp[i],dp[i-sq]+1)
        return dp[-1]
```

### 873. Length of Longest Fibonacci Subsequence

A sequence `X_1, X_2, ..., X_n` is *fibonacci-like* if:

- `n >= 3`
- `X_i + X_{i+1} = X_{i+2}` for all `i + 2 <= n`

Given a **strictly increasing** array `A` of positive integers forming a sequence, find the **length** of the longest fibonacci-like subsequence of `A`.  If one does not exist, return 0.

(*Recall that a subsequence is derived from another sequence A by deleting any number of elements (including none) from A, without changing the order of the remaining elements.  For example, [3, 5, 8] is a subsequence of [3, 4, 5, 6, 7, 8].*)

**Example 1:**

```
Input: [1,2,3,4,5,6,7,8]
Output: 5
Explanation:
The longest subsequence that is fibonacci-like: [1,2,3,5,8].
```

**Example 2:**

```
Input: [1,3,7,11,12,14,18]
Output: 3
Explanation:
The longest subsequence that is fibonacci-like:
[1,11,12], [3,11,14] or [7,11,18].
```

**Note:**

- `3 <= A.length <= 1000`
- `1 <= A[0] < A[1] < ... < A[A.length - 1] <= 10^9`

**状态方程**

```python
dp[A[j], A[i]] = dp.get((A[i] - A[j], A[j]), 2) + 1
能组成斐波那契数列的长度至少为3，dp[a,b]为上一个组成斐波那契数列的长度+1，即dp[b-a,a]+1
```

**Answer**

```python
class Solution:
    def lenLongestFibSubseq(self, A: List[int]) -> int:
        dp ={}
        s = set(A)
        for i in range(len(A)):
            for j in range(i):
                if A[i] - A[j] <A[j] and A[i]-A[j] in s:
                    dp[A[j], A[i]] = dp.get((A[i] - A[j], A[j]), 2) + 1
        return max(dp.values() or [0])
```

### 322. Coin Change

You are given coins of different denominations and a total amount of money *amount*. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`.

**Example 1:**

```
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
```

**Example 2:**

```
Input: coins = [2], amount = 3
Output: -1
```

**Note**:
You may assume that you have an infinite number of each kind of coin.

**状态方程**

```python
rs[i] = min(rs[i], rs[i-c] + 1)
假设金币最小面额为1，amount最大为1的倍数，向前寻找优化最小面额
```

**Answer**

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        rs = [amount+1] * (amount+1)
        rs[0] = 0
        for i in range(1, amount+1):
            for c in coins:
                if i >= c:
                    rs[i] = min(rs[i], rs[i-c] + 1)
        if rs[amount] == amount+1:
            return -1
        return rs[amount]
```



### 1024. Video Stitching

Medium

You are given a series of video clips from a sporting event that lasted `T` seconds.  These video clips can be overlapping with each other and have varied lengths.

Each video clip `clips[i]` is an interval: it starts at time `clips[i][0]` and ends at time `clips[i][1]`.  We can cut these clips into segments freely: for example, a clip `[0, 7]` can be cut into segments `[0, 1] + [1, 3] + [3, 7]`.

Return the minimum number of clips needed so that we can cut the clips into segments that cover the entire sporting event (`[0, T]`).  If the task is impossible, return `-1`.

 

**Example 1:**

```
Input: clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10
Output: 3
Explanation: 
We take the clips [0,2], [8,10], [1,9]; a total of 3 clips.
Then, we can reconstruct the sporting event as follows:
We cut [1,9] into segments [1,2] + [2,8] + [8,9].
Now we have segments [0,2] + [2,8] + [8,10] which cover the sporting event [0, 10].
```

**Example 2:**

```
Input: clips = [[0,1],[1,2]], T = 5
Output: -1
Explanation: 
We can't cover [0,5] with only [0,1] and [0,2].
```

**Example 3:**

```
Input: clips = [[0,1],[6,8],[0,2],[5,6],[0,4],[0,3],[6,7],[1,3],[4,7],[1,4],[2,5],[2,6],[3,4],[4,5],[5,7],[6,9]], T = 9
Output: 3
Explanation: 
We can take clips [0,4], [4,7], and [6,9].
```

**Example 4:**

```
Input: clips = [[0,4],[2,8]], T = 5
Output: 2
Explanation: 
Notice you can have extra video after the event ends.
```

 

**Note:**

1. `1 <= clips.length <= 100`
2. `0 <= clips[i][0], clips[i][1] <= 100`
3. `0 <= T <= 100`

**Answer**

```python
class Solution:
    def videoStitching(self, clips: List[List[int]], T: int) -> int:
        dp = [T+1 for _ in range(T+1)]
        dp[0] = 0
        for i in range(T+1):
            for c in clips:
                if c[0]<=i and c[1]>=i:
                    dp[i] =  min(dp[i],dp[c[0]]+1)
        return dp[T] if dp[T] != T+1 else -1
```



### 139. Word Break

Medium

Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words, determine if *s* can be segmented into a space-separated sequence of one or more dictionary words.

**Note:**

- The same word in the dictionary may be reused multiple times in the segmentation.
- You may assume the dictionary does not contain duplicate words.

**Example 1:**

```
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```

**Example 2:**

```
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
```

**Example 3:**

```
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
```

**Answer**

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False for _ in range(len(s)+1)]
        dp[0] = True
        for i in range(len(dp)):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        return dp[-1]
```



### 650. 2 Keys Keyboard

Medium

Initially on a notepad only one character 'A' is present. You can perform two operations on this notepad for each step:

1. `Copy All`: You can copy all the characters present on the notepad (partial copy is not allowed).
2. `Paste`: You can paste the characters which are copied **last time**.

 

Given a number `n`. You have to get **exactly** `n` 'A' on the notepad by performing the minimum number of steps permitted. Output the minimum number of steps to get `n` 'A'.

**Example 1:**

```
Input: 3
Output: 3
Explanation:
Intitally, we have one character 'A'.
In step 1, we use Copy All operation.
In step 2, we use Paste operation to get 'AA'.
In step 3, we use Paste operation to get 'AAA'.
```

 

**Note:**

1. The `n` will be in the range [1, 1000].

**Answer**

```python
class Solution:
    def minSteps(self, n: int) -> int:
        if n == 1: return 0
        dp = [n for _ in range(n+1)]
        dp[0] = 0
        dp[1] = 0
        dp[2] = 2
        for i in range(2,n+1):
            for j in range(1,i):
                if i%j!=0:
                    continue
                dp[i] = min(dp[i],int(dp[j]+i/j))
        return dp[n]
```







## 回溯法

### 22. Generate Parentheses

Given *n* pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given *n* = 3, a solution set is:

```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

**解释**

统计左括号的数目，如果左括号数小于pair 数，则可以添加左括号，当右括号小于左括号数时，则可以添加右括号，利用递归计算结果，递归到底时记录结果

**Answer**

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        answer = []
        def backtrack(s,left,right):
            if len(s) == 2*n:
                answer.append(s)
                return
            if left < n:
                backtrack(s+"(",left+1,right)
            if right < left:
                backtrack(s+")",left,right+1)
        
        backtrack("",0,0)
        return answer
```

### 46. Permutations

Given a collection of **distinct** integers, return all possible permutations.

**Example:**

```
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

**Answer**

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        answer = []
        def backtrack(start,end):
            if start == end:
                answer.append(nums[:])
                return
            for i in range(start,end):
                nums[start],nums[i] = nums[i],nums[start]
                backtrack(start+1,end)
                nums[start],nums[i] = nums[i],nums[start]
                
        def backtrack2(nums,path):
            if not nums:
                answer.append(path)
                return
            for i in range(len(nums)):
                backtrack2(nums[:i]+nums[i+1:],path+[nums[i]])
        # backtrack(0,len(nums))
        backtrack2(nums,[])
        return answer
```

### 17. Letter Combinations of a Phone Number

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Telephone-keypad2.svg/200px-Telephone-keypad2.svg.png)

**Example:**

```
Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

**Answer**

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {"2":'abc',"3":"def","4":"ghi",
              "5":"jkl","6":"mno","7":"pqrs",
               "8":"tuv","9":"wxyz"
              }
        if not digits:
            return []
        def backtrack(nums,path):
            if len(path)==len(digits):
                answer.append(path)
                return
            for i in range(len(nums)):
                for j in range(len(dic[nums[i]])):
                    backtrack(nums[i+1:],path+str(dic[nums[i]][j]))
        answer = []
        backtrack(digits,"")
        return answer
```

### 39. Combination Sum

Given a **set** of candidate numbers (`candidates`) **(without duplicates)** and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

The **same** repeated number may be chosen from `candidates` unlimited number of times.

**Note:**

- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```

**Example 2:**

```
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        answer = []
        def helper(nums,path):
            current_sum = sum(path)
            if current_sum == target:
                answer.append(path)
                return
            for i in range(len(nums)):
                if current_sum + nums[i] <= target:
                    helper(nums[i:],path+[nums[i]])
        for i in range(len(candidates)):
            helper(candidates[i:],[candidates[i]])
        return answer
```

