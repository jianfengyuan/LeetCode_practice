class BinarySearch:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        right = len(nums)-1
        left = 0
        
        while right >= left:
            mid = (right+left)//2
            if target > nums[mid]:
                left = mid+1
            elif target < nums[mid]:
                right = mid-1
            else:
                return target
        return -1

    def search_recursive(self,nums,target):
        right = len(nums)-1
        left = 0
        mid = right//2
        if left == right:
            if nums[0] != target:
                return -1
        if target > nums[mid]:
            return self.search_recursive(nums[mid+1:],target)
        elif target < nums[mid]:
            return self.search_recursive(nums[:mid],target)
        elif target == nums[mid]:
            return target



b = BinarySearch()
print(b.search_recursive([-1,0,3,5,9,12],9))