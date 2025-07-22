---
title: Two Pointers
description: Java solutions with explanations, time and space complexity for Two Pointers problems.
date: "June 1 2025"
---

# Two Pointers

## 1. Valid Palindrome

**Description:**  
Given a string `s`, return `true` if it is a palindrome, considering only alphanumeric characters and ignoring cases.

**Java Solution:**
```java
class Solution {
    public boolean isPalindrome(String s) {
        if (s.isEmpty()) {
            return true;
        }
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            char leftChar = s.charAt(left);
            char rightChar = s.charAt(right);

            if (!Character.isLetterOrDigit(leftChar)) {
                left++;
            } else if (!Character.isLetterOrDigit(rightChar)) {
                right--;
            } else {
                if (Character.toLowerCase(leftChar) != Character.toLowerCase(rightChar)) {
                    return false;
                }
                left++;
                right--;
            }
        }
        return true;
    }
}
```

**Explanation:**  
We use two pointers, one starting from the beginning and one from the end. We move them inwards, skipping non-alphanumeric characters. We compare characters in a case-insensitive manner.

- **Time Complexity:** O(n)  
  We iterate through the string with two pointers, each pointer traversing at most n characters.
- **Space Complexity:** O(1)  
  No extra space is used beyond a few variables.

---

## 2. Two Sum II - Input Array Is Sorted

**Description:**  
Given a 1-indexed array of integers `numbers` that is already sorted in non-decreasing order, find two numbers such that they add up to a specific `target` number.

**Java Solution:**
```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;

        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[] { left + 1, right + 1 };
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        return new int[] {}; // Should not reach here as per problem constraints
    }
}
```

**Explanation:**  
Since the array is sorted, we can use two pointers. If the sum is too small, increment `left`; if too large, decrement `right`.

- **Time Complexity:** O(n)  
  The two pointers traverse the array at most once, performing constant time operations per step.
- **Space Complexity:** O(1)  
  Only a few variables are used.

---

## 3. 3Sum

**Description:**  
Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

**Java Solution:**
```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue; // Skip duplicates

            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) left++; // Skip duplicates
                    while (left < right && nums[right] == nums[right - 1]) right--; // Skip duplicates
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }
}
```

**Explanation:**  
We sort the array first. Then, for each element, we use two pointers on the remaining array to find pairs that sum to the negative of the current element. Duplicates are handled to avoid redundant triplets.

- **Time Complexity:** O(n^2)  
  Sorting takes O(n log n). The nested loop with two pointers takes O(n^2). The dominant factor is O(n^2).
- **Space Complexity:** O(log n) to O(n)  
  This depends on the sorting algorithm used (e.g., quicksort uses O(log n) space for recursion stack; mergesort uses O(n)). Ignoring the output list.

---

## 4. Container With Most Water

**Description:**  
Given `n` non-negative integers `a1, a2, ..., an`, where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which, together with the x-axis, forms a container, such that the container contains the most water.

**Java Solution:**
```java
class Solution {
    public int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0;
        int right = height.length - 1;

        while (left < right) {
            int currentArea = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, currentArea);

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }
}
```

**Explanation:**  
We use two pointers, starting at the ends. At each step, we calculate the area. We move the pointer from the shorter line inwards, as moving the taller line's pointer would not increase the height and would only decrease the width.

- **Time Complexity:** O(n)  
  The two pointers traverse the array at most once.
- **Space Complexity:** O(1)  
  Only a few variables are used.

---

## 5. Trapping Rain Water

**Description:**  
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Java Solution:**
```java
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length == 0) return 0;

        int left = 0;
        int right = height.length - 1;
        int leftMax = 0;
        int rightMax = 0;
        int trappedWater = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    trappedWater += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    trappedWater += rightMax - height[right];
                }
                right--;
            }
        }
        return trappedWater;
    }
}
```

**Explanation:**  
We use two pointers, `left` and `right`, and keep track of the `leftMax` and `rightMax` heights encountered so far. At each step, we move the pointer from the shorter side. The amount of water trapped at a position is determined by the minimum of `leftMax` and `rightMax` minus the current bar's height.

- **Time Complexity:** O(n)  
  The two pointers traverse the array at most once.
- **Space Complexity:** O(1)  
  Only a few variables are used.

--- 