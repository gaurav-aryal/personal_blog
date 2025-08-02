---
title: Binary Search
description: Java solutions with explanations, time and space complexity for Binary Search problems from Blind 75.
date: "June 1 2025"
order: 5
---

# Binary Search

This section covers problems that can be efficiently solved using binary search. These problems often involve searching in sorted arrays or finding optimal values in a range.

## 1. Find Minimum in Rotated Sorted Array (Medium)

**Problem:** Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:
- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Given the sorted rotated array `nums` of unique elements, return the minimum element of this array.

**Example:**
```
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
```

**Solution:**
```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] > nums[right]) {
                // Minimum is in the right half
                left = mid + 1;
            } else {
                // Minimum is in the left half (including mid)
                right = mid;
            }
        }
        
        return nums[left];
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

**Key Insight:** The minimum element is always in the half that contains the smaller values.

---

## 2. Search in Rotated Sorted Array (Medium)

**Problem:** There is an integer array `nums` sorted in ascending order (with distinct values).

Prior to being passed to your function, `nums` is possibly rotated at an unknown pivot index `k` (`1 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed).

Given the array `nums` after the possible rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`.

**Example:**
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Solution:**
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            }
            
            // Check if left half is sorted
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

**Key Insight:** At least one half of the array is always sorted. Use this to determine which half to search.

---

## 3. Search in Rotated Sorted Array II (Medium)

**Problem:** There is an integer array `nums` sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, `nums` is rotated at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed).

Given the array `nums` after the possible rotation and an integer `target`, return `true` if `target` is in `nums`, or `false` if it is not in `nums`.

**Example:**
```
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
```

**Solution:**
```java
class Solution {
    public boolean search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return true;
            }
            
            // Handle duplicates
            if (nums[left] == nums[mid] && nums[mid] == nums[right]) {
                left++;
                right--;
                continue;
            }
            
            // Check if left half is sorted
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return false;
    }
}
```

**Time Complexity:** O(n) in worst case due to duplicates
**Space Complexity:** O(1)

**Key Insight:** When duplicates exist, we need to handle the case where we can't determine which half is sorted.

---

## 4. Find First and Last Position of Element in Sorted Array (Medium)

**Problem:** Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

**Example:**
```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Solution:**
```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1, -1};
        
        if (nums.length == 0) return result;
        
        // Find first occurrence
        result[0] = findFirst(nums, target);
        
        // Find last occurrence
        result[1] = findLast(nums, target);
        
        return result;
    }
    
    private int findFirst(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                result = mid;
                right = mid - 1; // Continue searching left
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    private int findLast(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                result = mid;
                left = mid + 1; // Continue searching right
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 5. Search Insert Position (Easy)

**Problem:** Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

**Example:**
```
Input: nums = [1,3,5,6], target = 5
Output: 2

Input: nums = [1,3,5,6], target = 2
Output: 1
```

**Solution:**
```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return left;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

**Key Insight:** When the loop ends, `left` points to the position where the target should be inserted.

---

## 6. Sqrt(x) (Easy)

**Problem:** Given a non-negative integer `x`, compute and return the square root of `x`.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

**Example:**
```
Input: x = 4
Output: 2

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.
```

**Solution:**
```java
class Solution {
    public int mySqrt(int x) {
        if (x == 0 || x == 1) return x;
        
        int left = 1, right = x;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (mid == x / mid) {
                return mid;
            } else if (mid < x / mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return right;
    }
}
```

**Time Complexity:** O(log x)
**Space Complexity:** O(1)

**Key Insight:** Use `mid == x / mid` instead of `mid * mid == x` to avoid integer overflow.

---

## 7. Valid Perfect Square (Easy)

**Problem:** Given a positive integer `num`, write a function which returns `true` if `num` is a perfect square else `false`.

**Example:**
```
Input: num = 16
Output: true

Input: num = 14
Output: false
```

**Solution:**
```java
class Solution {
    public boolean isPerfectSquare(int num) {
        if (num < 2) return true;
        
        int left = 2, right = num / 2;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            long square = (long) mid * mid;
            
            if (square == num) {
                return true;
            } else if (square < num) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return false;
    }
}
```

**Time Complexity:** O(log num)
**Space Complexity:** O(1)

**Key Insight:** Use `long` to avoid integer overflow when calculating the square.

## Key Takeaways

1. **Sorted Arrays**: Binary search works best on sorted arrays
2. **Pivot Point**: In rotated arrays, identify which half is sorted
3. **Duplicate Handling**: Be careful with duplicates in rotated arrays
4. **Boundary Conditions**: Always consider edge cases like empty arrays
5. **Overflow Prevention**: Use division instead of multiplication when possible
6. **Insert Position**: When target not found, `left` points to insertion position 