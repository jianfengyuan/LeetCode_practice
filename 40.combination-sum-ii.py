#
# @lc app=leetcode id=40 lang=python3
#
# [40] Combination Sum II
#
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def helper(nums,start,output,outputs,target):
            if target == 0:
                outputs.append(output)
                return
            for i in range(start,len(nums)):
                if i > start and nums[i] == nums[i-1]:
                    continue
                if nums[i] > target:
                    break
                helper(nums,i+1,output + [nums[i]],outputs,target-nums[i])
        outputs= []
        candidates.sort()
        helper(candidates,0,[],outputs,target)
        return outputs
