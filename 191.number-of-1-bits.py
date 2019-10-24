class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        """
        这里用一个trick， 可以轻松求出。 就是n & (n - 1) 可以消除 n 最后的一个1的原理。
        """
        if not n:  return 0
        count = 0
        while n:
            if n&1:
                count += 1
            n >>= 1
            # n &= (n-1)
            # count += 1
        return count 