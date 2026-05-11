from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l)//2 
            if nums[mid] == target:
                return mid
            if nums[mid] > nums[r]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:                                  # nums[mid] <= nums[r] → right half sorted
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
    
# Time complexity: O(log n)
# Space complexity: O(1)
# Example:
nums=[1,3]
target=3
print(Solution().search(nums, target)) # Output: 1