#
# @lc app=leetcode id=47 lang=python3
#
# [47] Permutations II
#
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # if not nums: return
        # def helper(nums,output, outputs):
        #     if not nums:
        #         if output not in outputs:
        #             outputs.append(output)
        #     for i in range(len(nums)):
        #         helper(nums[:i] + nums[i+1:], output + [nums[i]], outputs)
        # outputs= []
        # for i in range(len(nums)):
        #     helper(nums[:i]+nums[i+1:], [nums[i]], outputs)
        # return outputs
        if not nums:
            return []
        nums.sort()
        ret = [[]]
        for n in nums:
            new_ret = []
            l = len(ret[-1])
            for seq in ret:
                for i in range(l, -1, -1):
                    if i < l and seq[i] == n:
                        break
                    new_ret.append(seq[:i] + [n] + seq[i:])
            ret = new_ret
        return ret
