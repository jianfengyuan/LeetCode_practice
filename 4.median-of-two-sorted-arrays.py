#
# @lc app=leetcode id=4 lang=python3
#
# [4] Median of Two Sorted Arrays
#

# @lc code=start
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2) : return self.findMedianSortedArrays(nums2,nums1)
        n = len(nums1)
        m = len(nums2)
        lo = 0
        hi = n
        while lo > hi:
            i = (lo + hi)//2
            j = (m+n+1) // 2 - i
            maxLeftA = -float("INF") if i == 0 else nums1[i-1]
            minRightA = float("INF") if i == n else nums1[i]
            maxLeftB = -float("INF") if j == 0 else nums2[j-1]
            minRightB = float("INF") if j == m else nums2[j]
            print(minRightA,minRightB,maxLeftA,maxLeftB)
            if maxLeftA < minRightB and minRightA < maxLeftB:
                return (max(maxLeftA + maxLeftB) + max(minRightA+minRightB)) /2
            elif maxLeftA > minRightB:
                hi = i - 1
            else:
                lo = i + 1
            
        return 0.0
            # if i == 0:
            #     maxleftA = -float("INF")
            # else:
            #     maxleftA = nums1[i-1]

# @lc code=end

