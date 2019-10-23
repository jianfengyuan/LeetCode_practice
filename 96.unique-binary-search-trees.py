#
# @lc app=leetcode id=96 lang=python3
#
# [96] Unique Binary Search Trees
#
class Solution:
    def numTrees(self, n: int) -> int:
        '''
        卡特兰数
        把 n = 0 时赋为1，因为空树也算一种二叉搜索树，那么 n = 1 时的情况可以看做是其左子树个数乘以右子树的个数，
        左右子树都是空树，所以1乘1还是1。那么 n = 2 时，由于1和2都可以为根，分别算出来，再把它们加起来即可。
        n = 2 的情况可由下面式子算出（这里的 dp[i] 表示当有i个数字能组成的 BST 的个数）：

        dp[2] =  dp[0] * dp[1]　　　(1为根的情况，则左子树一定不存在，右子树可以有一个数字)

        　　　　+ dp[1] * dp[0]　　  (2为根的情况，则左子树可以有一个数字，右子树一定不存在)

        同理可写出 n = 3 的计算方法：

        dp[3] =  dp[0] * dp[2]　　　(1为根的情况，则左子树一定不存在，右子树可以有两个数字)

        　　　　+ dp[1] * dp[1]　　  (2为根的情况，则左右子树都可以各有一个数字)

        　　　  + dp[2] * dp[0]　　  (3为根的情况，则左子树可以有两个数字，右子树一定不存在)
        '''
        if not n: return 1
        dp  = [0]*(n+1)
        dp[0], dp[1] = 1, 1
        for i in range(2,n+1):
            for j in range(i):
                dp[i] += dp[j]*dp[i-j-1]
        return dp[-1]

