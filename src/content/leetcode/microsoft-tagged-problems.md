---
title: "Microsoft Tagged LeetCode Problems and Solutions"
description: "Comprehensive solutions for Microsoft tagged LeetCode problems with Java implementations, time and space complexity analysis."
date: "2025-01-27"
order: 24
---

# Microsoft Tagged LeetCode Problems and Solutions

This section covers Microsoft tagged LeetCode problems with detailed solutions, examples, and complexity analysis. These problems are commonly asked in Microsoft interviews and represent important algorithmic concepts.

## 1. Two Sum (Easy)

**Problem:** Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

**Example:**
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
```

**Solution:**
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        
        return new int[]{};
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 2. Climbing Stairs (Easy)

**Problem:** You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps.

**Example:**
```
Input: n = 3
Output: 3
```

**Solution:**
```java
class Solution {
    public int climbStairs(int n) {
        if (n <= 2) return n;
        
        int prev2 = 1;
        int prev1 = 2;
        
        for (int i = 3; i <= n; i++) {
            int current = prev1 + prev2;
            prev2 = prev1;
            prev1 = current;
        }
        
        return prev1;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 3. Container With Most Water (Medium)

**Problem:** Given n non-negative integers height where each represents a point at coordinate (i, height[i]), find two lines that together with the x-axis form a container that would hold the maximum amount of water.

**Example:**
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
```

**Solution:**
```java
class Solution {
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int maxArea = 0;
        
        while (left < right) {
            int width = right - left;
            int h = Math.min(height[left], height[right]);
            maxArea = Math.max(maxArea, width * h);
            
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

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 4. Search In Rotated Sorted Array (Medium)

**Problem:** There is an integer array nums sorted in ascending order (with distinct values).

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
            
            if (nums[mid] == target) return mid;
            
            if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
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

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 5. Longest Substring Without Repeating Characters (Medium)

**Problem:** Given a string s, find the length of the longest substring without repeating characters.

**Example:**
```
Input: s = "abcabcbb"
Output: 3
```

**Solution:**
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) return 0;
        
        Map<Character, Integer> map = new HashMap<>();
        int maxLength = 0;
        int start = 0;
        
        for (int end = 0; end < s.length(); end++) {
            char c = s.charAt(end);
            
            if (map.containsKey(c)) {
                start = Math.max(start, map.get(c) + 1);
            }
            
            map.put(c, end);
            maxLength = Math.max(maxLength, end - start + 1);
        }
        
        return maxLength;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(min(m, n))

---

## 6. 3Sum (Medium)

**Problem:** Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

**Example:**
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

**Solution:**
```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            
            int left = i + 1, right = nums.length - 1;
            
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                
                if (sum == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    
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

**Time Complexity:** O(n²)
**Space Complexity:** O(1)

---

## 7. Find The Index of The First Occurrence in a String (Easy)

**Problem:** Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

**Example:**
```
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
```

**Solution:**
```java
class Solution {
    public int strStr(String haystack, String needle) {
        if (needle.isEmpty()) return 0;
        if (haystack.length() < needle.length()) return -1;
        
        for (int i = 0; i <= haystack.length() - needle.length(); i++) {
            if (haystack.substring(i, i + needle.length()).equals(needle)) {
                return i;
            }
        }
        
        return -1;
    }
}
```

**Time Complexity:** O((n - m) × m)
**Space Complexity:** O(1)

---

## 8. Jump Game II (Medium)

**Problem:** Given an array of non-negative integers nums, you are initially positioned at the first index of the array.

**Example:**
```
Input: nums = [2,3,1,1,4]
Output: 2
```

**Solution:**
```java
class Solution {
    public int jump(int[] nums) {
        if (nums == null || nums.length < 2) return 0;
        
        int jumps = 0;
        int currentEnd = 0;
        int currentFarthest = 0;
        
        for (int i = 0; i < nums.length - 1; i++) {
            currentFarthest = Math.max(currentFarthest, i + nums[i]);
            
            if (i == currentEnd) {
                jumps++;
                currentEnd = currentFarthest;
            }
        }
        
        return jumps;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 9. Rotate Image (Medium)

**Problem:** You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

**Example:**
```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
```

**Solution:**
```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        
        // Transpose
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        // Reverse each row
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = temp;
            }
        }
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)

---

## 10. Minimum Size Subarray Sum (Medium)

**Problem:** Given an array of positive integers nums and a positive integer target, return the minimal length of a subarray whose sum is greater than or equal to target.

**Example:**
```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
```

**Solution:**
```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0, sum = 0;
        int minLen = Integer.MAX_VALUE;
        
        for (int right = 0; right < nums.length; right++) {
            sum += nums[right];
            
            while (sum >= target) {
                minLen = Math.min(minLen, right - left + 1);
                sum -= nums[left];
                left++;
            }
        }
        
        return minLen == Integer.MAX_VALUE ? 0 : minLen;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 11. Reverse Integer (Medium)

**Problem:** Given a signed 32-bit integer x, return x with its digits reversed.

**Example:**
```
Input: x = 123
Output: 321
```

**Solution:**
```java
class Solution {
    public int reverse(int x) {
        int result = 0;
        
        while (x != 0) {
            int digit = x % 10;
            
            if (result > Integer.MAX_VALUE / 10 || 
                (result == Integer.MAX_VALUE / 10 && digit > 7)) {
                return 0;
            }
            
            if (result < Integer.MIN_VALUE / 10 || 
                (result == Integer.MIN_VALUE / 10 && digit < -8)) {
                return 0;
            }
            
            result = result * 10 + digit;
            x /= 10;
        }
        
        return result;
    }
}
```

**Time Complexity:** O(log x)
**Space Complexity:** O(1)

---

## 12. Roman to Integer (Easy)

**Problem:** Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

**Example:**
```
Input: s = "III"
Output: 3
```

**Solution:**
```java
class Solution {
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        
        int result = 0;
        int prev = 0;
        
        for (int i = s.length() - 1; i >= 0; i--) {
            int current = map.get(s.charAt(i));
            if (current >= prev) {
                result += current;
            } else {
                result -= current;
            }
            prev = current;
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 13. Reverse Nodes In K Group (Hard)

**Problem:** Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

**Example:**
```
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
```

**Solution:**
```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        int count = 0;
        ListNode current = head;
        
        while (current != null) {
            count++;
            if (count % k == 0) {
                prev = reverse(prev, current.next);
                current = prev.next;
            } else {
                current = current.next;
            }
        }
        
        return dummy.next;
    }
    
    private ListNode reverse(ListNode prev, ListNode next) {
        ListNode last = prev.next;
        ListNode current = last.next;
        
        while (current != next) {
            last.next = current.next;
            current.next = prev.next;
            prev.next = current;
            current = last.next;
        }
        
        return last;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 14. Best Time to Buy And Sell Stock (Easy)

**Problem:** You are given an array prices where prices[i] is the price of a given stock on the ith day.

**Example:**
```
Input: prices = [7,1,5,3,6,4]
Output: 5
```

**Solution:**
```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length < 2) return 0;
        
        int minPrice = prices[0];
        int maxProfit = 0;
        
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else {
                maxProfit = Math.max(maxProfit, prices[i] - minPrice);
            }
        }
        
        return maxProfit;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 15. K-th Smallest in Lexicographical Order (Hard)

**Problem:** Given two integers n and k, return the kth lexicographically smallest integer in the range [1, n].

**Example:**
```
Input: n = 13, k = 2
Output: 10
```

**Solution:**
```java
class Solution {
    public int findKthNumber(int n, int k) {
        long curr = 1;
        k--;
        
        while (k > 0) {
            long steps = getSteps(n, curr, curr + 1);
            
            if (steps <= k) {
                curr++;
                k -= steps;
            } else {
                curr *= 10;
                k--;
            }
        }
        
        return (int) curr;
    }
    
    private long getSteps(int n, long curr, long next) {
        long steps = 0;
        
        while (curr <= n) {
            steps += Math.min(n + 1, next) - curr;
            curr *= 10;
            next *= 10;
        }
        
        return steps;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 16. Frequency of The Most Frequent Element (Medium)

**Problem:** The frequency of an element is the number of times it occurs in an array.

**Example:**
```
Input: nums = [1,2,4], k = 5
Output: 3
```

**Solution:**
```java
class Solution {
    public int maxFrequency(int[] nums, int k) {
        Arrays.sort(nums);
        int left = 0, right = 0;
        long sum = 0;
        int maxFreq = 0;
        
        while (right < nums.length) {
            sum += nums[right];
            
            while ((long) nums[right] * (right - left + 1) - sum > k) {
                sum -= nums[left];
                left++;
            }
            
            maxFreq = Math.max(maxFreq, right - left + 1);
            right++;
        }
        
        return maxFreq;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(1)

---

## 17. Max Consecutive Ones (Easy)

**Problem:** Given a binary array nums, return the maximum number of consecutive 1's in the array.

**Example:**
```
Input: nums = [1,1,0,1,1,1]
Output: 3
```

**Solution:**
```java
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int maxCount = 0;
        int currentCount = 0;
        
        for (int num : nums) {
            if (num == 1) {
                currentCount++;
                maxCount = Math.max(maxCount, currentCount);
            } else {
                currentCount = 0;
            }
        }
        
        return maxCount;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 18. Kth Smallest Product of Two Sorted Arrays (Hard)

**Problem:** Given two sorted 0-indexed integer arrays nums1 and nums2 as well as an integer k, return the kth (1-based) smallest product of nums1[i] * nums2[j] where 0 <= i < nums1.length and 0 <= j < nums2.length.

**Example:**
```
Input: nums1 = [2,5], nums2 = [3,4], k = 2
Output: 8
```

**Solution:**
```java
class Solution {
    public long kthSmallestProduct(int[] nums1, int[] nums2, long k) {
        long left = (long) -1e10, right = (long) 1e10;
        
        while (left < right) {
            long mid = left + (right - left) / 2;
            
            if (countLessOrEqual(nums1, nums2, mid) >= k) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return left;
    }
    
    private long countLessOrEqual(int[] nums1, int[] nums2, long target) {
        long count = 0;
        
        for (int num1 : nums1) {
            if (num1 == 0) {
                if (target >= 0) count += nums2.length;
            } else if (num1 > 0) {
                count += countLessOrEqualInArray(nums2, target / num1);
            } else {
                count += countGreaterOrEqualInArray(nums2, target / num1);
            }
        }
        
        return count;
    }
    
    private int countLessOrEqualInArray(int[] arr, long target) {
        int left = 0, right = arr.length;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }
    
    private int countGreaterOrEqualInArray(int[] arr, long target) {
        int left = 0, right = arr.length;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return arr.length - left;
    }
}
```

**Time Complexity:** O(n log n log(max))
**Space Complexity:** O(1)

---

## 19. Maximum Difference Between Even and Odd Frequency I (Easy)

**Problem:** You are given a 0-indexed integer array nums. A subarray is a contiguous non-empty sequence of elements within an array.

**Example:**
```
Input: nums = [1,2,3,4,5]
Output: 4
```

**Solution:**
```java
class Solution {
    public int maxFrequency(int[] nums, int k) {
        Arrays.sort(nums);
        int left = 0, right = 0;
        long sum = 0;
        int maxFreq = 0;
        
        while (right < nums.length) {
            sum += nums[right];
            
            while ((long) nums[right] * (right - left + 1) - sum > k) {
                sum -= nums[left];
                left++;
            }
            
            maxFreq = Math.max(maxFreq, right - left + 1);
            right++;
        }
        
        return maxFreq;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(1)

---

## 20. Next Permutation (Medium)

**Problem:** Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

**Example:**
```
Input: nums = [1,2,3]
Output: [1,3,2]
```

**Solution:**
```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1) return;
        
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        
        reverse(nums, i + 1);
    }
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    
    private void reverse(int[] nums, int start) {
        int end = nums.length - 1;
        while (start < end) {
            swap(nums, start, end);
            start++;
            end--;
        }
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 21. Find First And Last Position of Element In Sorted Array (Medium)

**Problem:** Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

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
        result[0] = findFirst(nums, target);
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
                right = mid - 1;
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
                left = mid + 1;
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

## 22. Recover Binary Search Tree (Hard)

**Problem:** You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake.

**Example:**
```
Input: root = [1,3,null,null,2]
Output: [3,1,null,null,2]
```

**Solution:**
```java
class Solution {
    private TreeNode first = null;
    private TreeNode second = null;
    private TreeNode prev = new TreeNode(Integer.MIN_VALUE);
    
    public void recoverTree(TreeNode root) {
        inorder(root);
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }
    
    private void inorder(TreeNode root) {
        if (root == null) return;
        inorder(root.left);
        if (first == null && prev.val > root.val) {
            first = prev;
        }
        if (first != null && prev.val > root.val) {
            second = root;
        }
        prev = root;
        inorder(root.right);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 23. Merge Sorted Array (Easy)

**Problem:** You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

**Example:**
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```

**Solution:**
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1;
        int p2 = n - 1;
        int p = m + n - 1;
        
        while (p1 >= 0 && p2 >= 0) {
            if (nums1[p1] > nums2[p2]) {
                nums1[p] = nums1[p1];
                p1--;
            } else {
                nums1[p] = nums2[p2];
                p2--;
            }
            p--;
        }
        
        while (p2 >= 0) {
            nums1[p] = nums2[p2];
            p2--;
            p--;
        }
    }
}
```

**Time Complexity:** O(m + n)
**Space Complexity:** O(1)

---

## 24. Add Two Numbers (Medium)

**Problem:** You are given two non-empty linked lists representing two non-negative integers.

**Example:**
```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
```

**Solution:**
```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        int carry = 0;
        
        while (l1 != null || l2 != null || carry != 0) {
            int sum = carry;
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            current.next = new ListNode(sum % 10);
            current = current.next;
            carry = sum / 10;
        }
        return dummy.next;
    }
}
```

**Time Complexity:** O(max(m, n))
**Space Complexity:** O(max(m, n))

---

## 25. Median of Two Sorted Arrays (Hard)

**Problem:** Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

**Example:**
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
```

**Solution:**
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int x = nums1.length;
        int y = nums2.length;
        int low = 0, high = x;
        
        while (low <= high) {
            int partitionX = (low + high) / 2;
            int partitionY = (x + y + 1) / 2 - partitionX;
            
            int maxLeftX = (partitionX == 0) ? Integer.MIN_VALUE : nums1[partitionX - 1];
            int minRightX = (partitionX == x) ? Integer.MAX_VALUE : nums1[partitionX];
            int maxLeftY = (partitionY == 0) ? Integer.MIN_VALUE : nums2[partitionY - 1];
            int minRightY = (partitionY == y) ? Integer.MAX_VALUE : nums2[partitionY];
            
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                if ((x + y) % 2 == 0) {
                    return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2.0;
                } else {
                    return Math.max(maxLeftX, maxLeftY);
                }
            } else if (maxLeftX > minRightY) {
                high = partitionX - 1;
            } else {
                low = partitionX + 1;
            }
        }
        throw new IllegalArgumentException();
    }
}
```

**Time Complexity:** O(log(min(m, n)))
**Space Complexity:** O(1)

---

## 26. Koko Eating Bananas (Medium)

**Problem:** Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas.

**Example:**
```
Input: piles = [3,6,7,11], h = 8
Output: 4
```

**Solution:**
```java
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1;
        int right = Arrays.stream(piles).max().getAsInt();
        
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
    
    private boolean canEatAll(int[] piles, int h, int speed) {
        int hours = 0;
        for (int pile : piles) {
            hours += (pile + speed - 1) / speed;
        }
        return hours <= h;
    }
}
```

**Time Complexity:** O(n log M)
**Space Complexity:** O(1)

---

## 27. Generate Parentheses (Medium)

**Problem:** Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Example:**
```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

**Solution:**
```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        backtrack(result, "", 0, 0, n);
        return result;
    }
    
    private void backtrack(List<String> result, String current, int open, int close, int max) {
        if (current.length() == max * 2) {
            result.add(current);
            return;
        }
        if (open < max) {
            backtrack(result, current + "(", open + 1, close, max);
        }
        if (close < open) {
            backtrack(result, current + ")", open, close + 1, max);
        }
    }
}
```

**Time Complexity:** O(4^n / √n)
**Space Complexity:** O(4^n / √n)

---

## 28. Serialize And Deserialize Binary Tree (Hard)

**Problem:** Design an algorithm to serialize and deserialize a binary tree.

**Example:**
```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
```

**Solution:**
```java
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "null";
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }
    
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
        return deserializeHelper(queue);
    }
    
    private TreeNode deserializeHelper(Queue<String> queue) {
        String val = queue.poll();
        if (val.equals("null")) return null;
        TreeNode root = new TreeNode(Integer.parseInt(val));
        root.left = deserializeHelper(queue);
        root.right = deserializeHelper(queue);
        return root;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 29. Group Anagrams (Medium)

**Problem:** Given an array of strings strs, group the anagrams together.

**Example:**
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

**Solution:**
```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            map.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        
        return new ArrayList<>(map.values());
    }
}
```

**Time Complexity:** O(n × k log k)
**Space Complexity:** O(n × k)

---

## 30. Maximum Subarray (Medium)

**Problem:** Given an integer array nums, find the subarray with the largest sum, and return its sum.

**Example:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
```

**Solution:**
```java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        
        int maxSoFar = nums[0];
        int maxEndingHere = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1) 

---

## 31. Rotate Array (Medium)

**Problem:** Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

**Example:**
```
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
```

**Solution:**
```java
class Solution {
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    
    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 32. Search a 2D Matrix (Medium)

**Problem:** Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix.

**Example:**
```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
```

**Solution:**
```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) return false;
        
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

**Time Complexity:** O(log(m × n))
**Space Complexity:** O(1)

---

## 33. Longest Palindromic Substring (Medium)

**Problem:** Given a string s, return the longest palindromic substring in s.

**Example:**
```
Input: s = "babad"
Output: "bab"
```

**Solution:**
```java
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 2) return s;
        
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            
            if (len > maxLen) {
                start = i - (len - 1) / 2;
                maxLen = len;
            }
        }
        
        return s.substring(start, start + maxLen);
    }
    
    private int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)

---

## 34. Search Insert Position (Easy)

**Problem:** Given a sorted array of distinct integers and a target value, return the index if the target is found.

**Example:**
```
Input: nums = [1,3,5,6], target = 5
Output: 2
```

**Solution:**
```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 35. Combination Sum II (Medium)

**Problem:** Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

**Example:**
```
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: [[1,1,6],[1,2,5],[1,7],[2,6]]
```

**Solution:**
```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(candidates, target, 0, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrack(int[] candidates, int target, int start, List<Integer> current, List<List<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        if (target < 0) return;
        
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            
            current.add(candidates[i]);
            backtrack(candidates, target - candidates[i], i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }
}
```

**Time Complexity:** O(2^n)
**Space Complexity:** O(n)

---

## 36. Trapping Rain Water (Hard)

**Problem:** Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Example:**
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

**Solution:**
```java
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length < 3) return 0;
        
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int water = 0;
        
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    water += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    water += rightMax - height[right];
                }
                right--;
            }
        }
        
        return water;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 37. Merge Intervals (Medium)

**Problem:** Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals.

**Example:**
```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

**Solution:**
```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> result = new ArrayList<>();
        int[] current = intervals[0];
        
        for (int i = 1; i < intervals.length; i++) {
            if (current[1] >= intervals[i][0]) {
                current[1] = Math.max(current[1], intervals[i][1]);
            } else {
                result.add(current);
                current = intervals[i];
            }
        }
        
        result.add(current);
        
        return result.toArray(new int[result.size()][]);
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

## 38. Sqrt(x) (Easy)

**Problem:** Given a non-negative integer x, compute and return the square root of x.

**Example:**
```
Input: x = 4
Output: 2
```

**Solution:**
```java
class Solution {
    public int mySqrt(int x) {
        if (x == 0) return 0;
        if (x == 1) return 1;
        
        int left = 1, right = x;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (mid <= x / mid && (mid + 1) > x / (mid + 1)) {
                return mid;
            } else if (mid > x / mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return left;
    }
}
```

**Time Complexity:** O(log x)
**Space Complexity:** O(1)

---

## 39. Largest Rectangle In Histogram (Hard)

**Problem:** Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Example:**
```
Input: heights = [2,1,5,6,2,3]
Output: 10
```

**Solution:**
```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int i = 0;
        
        while (i < heights.length) {
            if (stack.isEmpty() || heights[stack.peek()] <= heights[i]) {
                stack.push(i);
                i++;
            } else {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
        }
        
        while (!stack.isEmpty()) {
            int height = heights[stack.pop()];
            int width = stack.isEmpty() ? i : i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        
        return maxArea;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 40. Binary Tree Level Order Traversal (Medium)

**Problem:** Given the root of a binary tree, return the level order traversal of its nodes' values.

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
```

**Solution:**
```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            result.add(currentLevel);
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n) 

---

## 41. Longest Consecutive Sequence (Hard)

**Problem:** Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

**Example:**
```
Input: nums = [100,4,200,1,3,2]
Output: 4
```

**Solution:**
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int maxLen = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int current = num;
                int streak = 1;
                while (set.contains(current + 1)) {
                    current++;
                    streak++;
                }
                maxLen = Math.max(maxLen, streak);
            }
        }
        return maxLen;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 42. Intersection of Two Linked Lists (Easy)

**Problem:** Given the heads of two singly linked lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection, return null.

**Example:**
```
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5]
Output: Intersected at '8'
```

**Solution:**
```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode a = headA, b = headB;
        while (a != b) {
            a = (a == null) ? headB : a.next;
            b = (b == null) ? headA : b.next;
        }
        return a;
    }
}
```

**Time Complexity:** O(m + n)
**Space Complexity:** O(1)

---

## 43. Gas Station (Medium)

**Problem:** There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

**Example:**
```
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
```

**Solution:**
```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int total = 0, curr = 0, start = 0;
        for (int i = 0; i < gas.length; i++) {
            total += gas[i] - cost[i];
            curr += gas[i] - cost[i];
            if (curr < 0) {
                start = i + 1;
                curr = 0;
            }
        }
        return total < 0 ? -1 : start;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 44. LRU Cache (Medium)

**Problem:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

**Example:**
```
Input: ["LRUCache","put","put","get","put","get","put","get","get","get"]
       [[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]
Output: [null,null,null,1,null,-1,null,-1,3,4]
```

**Solution:**
```java
class LRUCache {
    private final int capacity;
    private final Map<Integer, Node> map;
    private final Node head, tail;
    
    private static class Node {
        int key, value;
        Node prev, next;
        Node(int k, int v) { key = k; value = v; }
    }
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        remove(node);
        insert(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            remove(map.get(key));
        }
        if (map.size() == capacity) {
            remove(tail.prev);
        }
        insert(new Node(key, value));
    }
    
    private void remove(Node node) {
        map.remove(node.key);
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void insert(Node node) {
        map.put(node.key, node);
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
}
```

**Time Complexity:** O(1) for both get and put
**Space Complexity:** O(capacity)

---

## 45. Find Peak Element (Medium)

**Problem:** A peak element is an element that is strictly greater than its neighbors. Given an integer array nums, find a peak element, and return its index.

**Example:**
```
Input: nums = [1,2,3,1]
Output: 2
```

**Solution:**
```java
class Solution {
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 46. Contains Duplicate (Easy)

**Problem:** Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

**Example:**
```
Input: nums = [1,2,3,1]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) return true;
            set.add(num);
        }
        return false;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 47. Split Array Largest Sum (Hard)

**Problem:** Given an array nums which consists of non-negative integers and an integer m, you can split the array into m non-empty continuous subarrays. Write an algorithm to minimize the largest sum among these m subarrays.

**Example:**
```
Input: nums = [7,2,5,10,8], m = 2
Output: 18
```

**Solution:**
```java
class Solution {
    public int splitArray(int[] nums, int m) {
        int left = Arrays.stream(nums).max().getAsInt();
        int right = Arrays.stream(nums).sum();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (canSplit(nums, m, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
    private boolean canSplit(int[] nums, int m, int maxSum) {
        int count = 1, sum = 0;
        for (int num : nums) {
            sum += num;
            if (sum > maxSum) {
                sum = num;
                count++;
                if (count > m) return false;
            }
        }
        return true;
    }
}
```

**Time Complexity:** O(n log(sum - max))
**Space Complexity:** O(1)

---

## 48. Subarray Sum Equals K (Medium)

**Problem:** Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.

**Example:**
```
Input: nums = [1,1,1], k = 2
Output: 2
```

**Solution:**
```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0, count = 0;
        for (int num : nums) {
            sum += num;
            count += map.getOrDefault(sum - k, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return count;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 49. Convert BST to Greater Tree (Medium)

**Problem:** Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.

**Example:**
```
Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
```

**Solution:**
```java
class Solution {
    private int sum = 0;
    public TreeNode convertBST(TreeNode root) {
        traverse(root);
        return root;
    }
    private void traverse(TreeNode node) {
        if (node == null) return;
        traverse(node.right);
        sum += node.val;
        node.val = sum;
        traverse(node.left);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 50. String Compression (Medium)

**Problem:** Given an array of characters chars, compress it in-place. The length after compression must always be less than or equal to the original array.

**Example:**
```
Input: chars = ["a","a","b","b","c","c","c"]
Output: 6, chars = ["a","2","b","2","c","3"]
```

**Solution:**
```java
class Solution {
    public int compress(char[] chars) {
        int n = chars.length, write = 0, anchor = 0;
        for (int read = 0; read < n; read++) {
            if (read + 1 == n || chars[read] != chars[read + 1]) {
                chars[write++] = chars[anchor];
                if (read > anchor) {
                    for (char c : Integer.toString(read - anchor + 1).toCharArray()) {
                        chars[write++] = c;
                    }
                }
                anchor = read + 1;
            }
        }
        return write;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1) 