---
title: Binary Search
description: Java solutions with explanations, time and space complexity for Binary Search problems.
date: "June 1 2025"
---

# Binary Search Pattern

Binary Search is a powerful algorithm for finding elements in sorted arrays. It's particularly useful for:
- Finding elements in sorted arrays
- Finding boundaries in sorted arrays
- Optimizing problems with monotonic functions
- Searching in rotated arrays
- Finding median or kth element

## 1. Binary Search (Easy)

**Problem:** Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

**Solution:**
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        
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
        
        return -1;
    }
}
```

**Time Complexity:** O(log n) where n is the length of the array
**Space Complexity:** O(1)

## 2. Search a 2D Matrix (Medium)

**Problem:** Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:
- Integers in each row are sorted from left to right
- The first integer of each row is greater than the last integer of the previous row

**Solution:**
```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int left = 0;
        int right = m * n - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int row = mid / n;
            int col = mid % n;
            
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return false;
    }
}
```

**Time Complexity:** O(log(m*n)) where m and n are dimensions of the matrix
**Space Complexity:** O(1)

## 3. Koko Eating Bananas (Medium)

**Problem:** Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours. Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat from any other pile during this hour. Return the minimum integer k such that she can eat all the bananas within h hours.

**Solution:**
```java
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1;
        int right = 1;
        
        // Find maximum pile size
        for (int pile : piles) {
            right = Math.max(right, pile);
        }
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (canEatAll(piles, h, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return left;
    }
    
    private boolean canEatAll(int[] piles, int h, int k) {
        int hours = 0;
        for (int pile : piles) {
            hours += (pile + k - 1) / k; // Ceiling division
        }
        return hours <= h;
    }
}
```

**Time Complexity:** O(n log m) where n is number of piles and m is max pile size
**Space Complexity:** O(1)

## 4. Search in Rotated Sorted Array (Medium)

**Problem:** There is an integer array nums sorted in ascending order (with distinct values). Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k. Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

**Solution:**
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            }
            
            // Check if left half is sorted
            if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if (target > nums[mid] && target <= nums[right]) {
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

**Time Complexity:** O(log n) where n is the length of the array
**Space Complexity:** O(1)

## 5. Find Minimum in Rotated Sorted Array (Medium)

**Problem:** Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]. Given the sorted rotated array nums of unique elements, return the minimum element of this array.

**Solution:**
```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return nums[left];
    }
}
```

**Time Complexity:** O(log n) where n is the length of the array
**Space Complexity:** O(1)

## 6. Time Based Key-Value Store (Medium)

**Problem:** Design a time-based key-value data structure that can store multiple values for the same key at different timestamps and retrieve the key's value at a certain timestamp.

**Solution:**
```java
class TimeMap {
    private Map<String, List<Pair<Integer, String>>> map;
    
    public TimeMap() {
        map = new HashMap<>();
    }
    
    public void set(String key, String value, int timestamp) {
        if (!map.containsKey(key)) {
            map.put(key, new ArrayList<>());
        }
        map.get(key).add(new Pair<>(timestamp, value));
    }
    
    public String get(String key, int timestamp) {
        if (!map.containsKey(key)) {
            return "";
        }
        
        List<Pair<Integer, String>> list = map.get(key);
        int left = 0;
        int right = list.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (list.get(mid).getKey() == timestamp) {
                return list.get(mid).getValue();
            } else if (list.get(mid).getKey() < timestamp) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return right >= 0 ? list.get(right).getValue() : "";
    }
}
```

**Time Complexity:** 
- set: O(1)
- get: O(log n) where n is number of timestamps for the key
**Space Complexity:** O(n) where n is total number of operations

## 7. Median of Two Sorted Arrays (Hard)

**Problem:** Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

**Solution:**
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int left = (m + n + 1) / 2;
        int right = (m + n + 2) / 2;
        
        return (findKth(nums1, 0, nums2, 0, left) + 
                findKth(nums1, 0, nums2, 0, right)) / 2.0;
    }
    
    private int findKth(int[] nums1, int i, int[] nums2, int j, int k) {
        if (i >= nums1.length) {
            return nums2[j + k - 1];
        }
        if (j >= nums2.length) {
            return nums1[i + k - 1];
        }
        if (k == 1) {
            return Math.min(nums1[i], nums2[j]);
        }
        
        int midVal1 = (i + k/2 - 1 < nums1.length) ? nums1[i + k/2 - 1] : Integer.MAX_VALUE;
        int midVal2 = (j + k/2 - 1 < nums2.length) ? nums2[j + k/2 - 1] : Integer.MAX_VALUE;
        
        if (midVal1 < midVal2) {
            return findKth(nums1, i + k/2, nums2, j, k - k/2);
        } else {
            return findKth(nums1, i, nums2, j + k/2, k - k/2);
        }
    }
}
```

**Time Complexity:** O(log(m+n)) where m and n are lengths of input arrays
**Space Complexity:** O(log(m+n)) for recursion stack

## Key Takeaways

1. Binary Search is perfect for:
   - Finding elements in sorted arrays
   - Finding boundaries in sorted arrays
   - Optimizing problems with monotonic functions
   - Searching in rotated arrays

2. Common patterns:
   - Use left + (right - left) / 2 to avoid overflow
   - Consider edge cases (empty array, single element)
   - Think about what to return (index vs value)
   - Handle duplicates if present

3. Tips:
   - Always verify if the array is sorted
   - Consider if the array can be rotated
   - Think about what happens when element is not found
   - Use binary search for optimization problems 