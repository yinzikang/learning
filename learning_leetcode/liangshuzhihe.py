import time
nums = [3, 2, 4]
target = 6


def method_1(nums, target):
    records = dict()
    for idx, val in enumerate(nums):
        if target - val not in records:
            records[val] = idx
        else:
            return records[target - val], idx


def method_2(nums, target):
    for idx, val in enumerate(nums):
        if target - val in nums:
            idx_2 = nums.index(target - val)
            if idx_2 != idx:
                return idx, idx_2

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for idx, val in enumerate(nums):
            if target - val in nums:
                idx_2 = nums.index(target - val)
                if idx_2 != idx:
                    return idx, idx_2

time1 = time.time()
m1a1, m1a2 = method_1(nums, target)
# time.sleep(1)
time2 = time.time()
m2a1, m2a2 = method_2(nums, target)
# time.sleep(1)
time3 = time.time()
print(time3 + time1-2*time2)
a = Solution()
m3a1, m3a2 = a.twoSum(nums, target)


print(m1a1, m1a2)
print(m2a1, m2a2)
print(m3a1, m3a2)