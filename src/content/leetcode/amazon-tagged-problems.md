---
title: "Amazon Tagged LeetCode Problems and Solutions"
description: "Comprehensive solutions for Amazon tagged LeetCode problems with Java implementations, time and space complexity analysis."
date: "2025-01-27"
order: 101
---
This section covers Amazon tagged LeetCode problems with detailed solutions, examples, and complexity analysis. These problems are commonly asked in Amazon interviews and represent important algorithmic concepts.

## 1. LRU Cache (Medium)

**Problem:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

**Example:**
```
Input: ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
       [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output: [null, null, null, 1, null, -1, null, -1, 3, 4]
```

**Solution:**
```java
class LRUCache {
    private Map<Integer, Node> cache;
    private Node head, tail;
    private int capacity;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        cache = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            remove(node);
            add(node);
            return node.value;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            remove(cache.get(key));
        }
        
        Node newNode = new Node(key, value);
        cache.put(key, newNode);
        add(newNode);
        
        if (cache.size() > capacity) {
            Node lru = head.next;
            remove(lru);
            cache.remove(lru.key);
        }
    }
    
    private void add(Node node) {
        node.prev = tail.prev;
        node.next = tail;
        tail.prev.next = node;
        tail.prev = node;
    }
    
    private void remove(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    class Node {
        int key, value;
        Node prev, next;
        
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(capacity)

---

## 2. Reorganize String (Medium)

**Problem:** Given a string `s`, rearrange the characters of `s` so that any two adjacent characters are not the same.

**Example:**
```
Input: s = "aab"
Output: "aba"
```

**Solution:**
```java
class Solution {
    public String reorganizeString(String s) {
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        
        PriorityQueue<Character> pq = new PriorityQueue<>((a, b) -> count[b - 'a'] - count[a - 'a']);
        for (int i = 0; i < 26; i++) {
            if (count[i] > 0) {
                pq.offer((char) (i + 'a'));
            }
        }
        
        StringBuilder result = new StringBuilder();
        while (pq.size() >= 2) {
            char first = pq.poll();
            char second = pq.poll();
            
            result.append(first);
            result.append(second);
            
            count[first - 'a']--;
            count[second - 'a']--;
            
            if (count[first - 'a'] > 0) {
                pq.offer(first);
            }
            if (count[second - 'a'] > 0) {
                pq.offer(second);
            }
        }
        
        if (!pq.isEmpty()) {
            char last = pq.poll();
            if (count[last - 'a'] > 1) {
                return "";
            }
            result.append(last);
        }
        
        return result.toString();
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

---

## 3. Two Sum (Easy)

**Problem:** Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

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

## 4. Maximum Frequency After Subarray Operation (Medium)

**Problem:** You are given an array `nums` of positive integers and an integer `k`. You can perform the following operation at most `k` times.

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

## 5. Maximum Profit in Job Scheduling (Hard)

**Problem:** We have `n` jobs, where every job is scheduled to be done from `startTime[i]` to `endTime[i]`, obtaining a profit of `profit[i]`.

**Example:**
```
Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
Output: 120
```

**Solution:**
```java
class Solution {
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        int[][] jobs = new int[n][3];
        
        for (int i = 0; i < n; i++) {
            jobs[i] = new int[]{startTime[i], endTime[i], profit[i]};
        }
        
        Arrays.sort(jobs, (a, b) -> a[1] - b[1]);
        
        TreeMap<Integer, Integer> dp = new TreeMap<>();
        dp.put(0, 0);
        
        for (int[] job : jobs) {
            int curr = dp.floorEntry(job[0]).getValue() + job[2];
            if (curr > dp.lastEntry().getValue()) {
                dp.put(job[1], curr);
            }
        }
        
        return dp.lastEntry().getValue();
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

## 6. Number of Islands (Medium)

**Problem:** Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the number of islands.

**Example:**
```
Input: grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
Output: 1
```

**Solution:**
```java
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j);
                }
            }
        }
        
        return count;
    }
    
    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0') {
            return;
        }
        
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
}
```

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n)

---

## 7. Maximize Y-Sum by Picking a Triplet of Distinct X-Values (Hard)

**Problem:** You are given an array `nums` of positive integers and an integer `k`.

**Example:**
```
Input: nums = [1,2,3,4,5], k = 2
Output: 9
```

**Solution:**
```java
class Solution {
    public long maximumTripletValue(int[] nums) {
        int n = nums.length;
        long maxDiff = 0;
        int maxNum = 0;
        long result = 0;
        
        for (int i = 0; i < n; i++) {
            result = Math.max(result, (long) maxDiff * nums[i]);
            maxDiff = Math.max(maxDiff, maxNum - nums[i]);
            maxNum = Math.max(maxNum, nums[i]);
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 8. Longest Palindromic Substring (Medium)

**Problem:** Given a string `s`, return the longest palindromic substring in `s`.

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

## 9. Analyze User Website Visit Pattern (Medium)

**Problem:** You are given two string arrays `username` and `website` and an integer array `timestamp`.

**Example:**
```
Input: username = ["joe","joe","joe","james","james","james","james","mary","mary","mary"], timestamp = [1,2,3,4,5,6,7,8,9,10], website = ["home","about","career","home","cart","maps","home","home","about","career"]
Output: ["home","about","career"]
```

**Solution:**
```java
class Solution {
    public List<String> mostVisitedPattern(String[] username, int[] timestamp, String[] website) {
        Map<String, List<Pair<Integer, String>>> userVisits = new HashMap<>();
        
        for (int i = 0; i < username.length; i++) {
            userVisits.computeIfAbsent(username[i], k -> new ArrayList<>())
                      .add(new Pair<>(timestamp[i], website[i]));
        }
        
        Map<String, Integer> patternCount = new HashMap<>();
        
        for (List<Pair<Integer, String>> visits : userVisits.values()) {
            if (visits.size() < 3) continue;
            
            visits.sort((a, b) -> a.getKey() - b.getKey());
            Set<String> patterns = new HashSet<>();
            
            for (int i = 0; i < visits.size() - 2; i++) {
                for (int j = i + 1; j < visits.size() - 1; j++) {
                    for (int k = j + 1; k < visits.size(); k++) {
                        String pattern = visits.get(i).getValue() + "," + 
                                      visits.get(j).getValue() + "," + 
                                      visits.get(k).getValue();
                        patterns.add(pattern);
                    }
                }
            }
            
            for (String pattern : patterns) {
                patternCount.put(pattern, patternCount.getOrDefault(pattern, 0) + 1);
            }
        }
        
        String result = "";
        int maxCount = 0;
        
        for (Map.Entry<String, Integer> entry : patternCount.entrySet()) {
            if (entry.getValue() > maxCount || 
                (entry.getValue() == maxCount && entry.getKey().compareTo(result) < 0)) {
                maxCount = entry.getValue();
                result = entry.getKey();
            }
        }
        
        return Arrays.asList(result.split(","));
    }
}
```

**Time Complexity:** O(n³)
**Space Complexity:** O(n³)

---

## 10. House Robber (Medium)

**Problem:** You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed.

**Example:**
```
Input: nums = [1,2,3,1]
Output: 4
```

**Solution:**
```java
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        int prev1 = nums[0];
        int prev2 = Math.max(nums[0], nums[1]);
        
        for (int i = 2; i < nums.length; i++) {
            int current = Math.max(prev1 + nums[i], prev2);
            prev1 = prev2;
            prev2 = current;
        }
        
        return prev2;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 11. Trapping Rain Water (Hard)

**Problem:** Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Example:**
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

**Solution:**
```java
class Solution {
    public int trap(int[] height) {
        if (height.length < 3) return 0;
        
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

## 12. Find First And Last Position of Element In Sorted Array (Medium)

**Problem:** Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

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

## 13. Jump Game II (Medium)

**Problem:** Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

**Example:**
```
Input: nums = [2,3,1,1,4]
Output: 2
```

**Solution:**
```java
class Solution {
    public int jump(int[] nums) {
        int jumps = 0;
        int currentEnd = 0;
        int farthest = 0;
        
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);
            
            if (i == currentEnd) {
                jumps++;
                currentEnd = farthest;
            }
        }
        
        return jumps;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 14. Put Marbles in Bags (Hard)

**Problem:** You have `k` bags. You are given a 0-indexed integer array `weights` where `weights[i]` is the weight of the `i`th marble.

**Example:**
```
Input: weights = [1,3,5,1], k = 2
Output: 4
```

**Solution:**
```java
class Solution {
    public long putMarbles(int[] weights, int k) {
        if (k == 1) return 0;
        
        int n = weights.length;
        int[] pairs = new int[n - 1];
        
        for (int i = 0; i < n - 1; i++) {
            pairs[i] = weights[i] + weights[i + 1];
        }
        
        Arrays.sort(pairs);
        
        long minSum = 0, maxSum = 0;
        for (int i = 0; i < k - 1; i++) {
            minSum += pairs[i];
            maxSum += pairs[n - 2 - i];
        }
        
        return maxSum - minSum;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

## 15. Median of Two Sorted Arrays (Hard)

**Problem:** Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.

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

## 16. Merge Intervals (Medium)

**Problem:** Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals.

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
        
        for (int[] interval : intervals) {
            if (current[1] >= interval[0]) {
                current[1] = Math.max(current[1], interval[1]);
            } else {
                result.add(current);
                current = interval;
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

## 17. Top K Frequent Elements (Medium)

**Problem:** Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

**Example:**
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

**Solution:**
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> pq = 
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        
        for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.poll().getKey();
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(n)

---

## 18. Best Time to Buy and Sell Stock (Easy)

**Problem:** You are given an array `prices` where `prices[i]` is the price of a given stock on the `i`th day.

**Example:**
```
Input: prices = [7,1,5,3,6,4]
Output: 5
```

**Solution:**
```java
class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        
        for (int price : prices) {
            minPrice = Math.min(minPrice, price);
            maxProfit = Math.max(maxProfit, price - minPrice);
        }
        
        return maxProfit;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 19. Add Two Numbers (Medium)

**Problem:** You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit.

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

## 20. Longest Substring Without Repeating Characters (Medium)

**Problem:** Given a string `s`, find the length of the longest substring without repeating characters.

**Example:**
```
Input: s = "abcabcbb"
Output: 3
```

**Solution:**
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, maxLength = 0;
        
        for (int right = 0; right < s.length(); right++) {
            while (set.contains(s.charAt(right))) {
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(min(m, n))

---

## 21. Rotting Oranges (Medium)

**Problem:** You are given an `m x n` grid where each cell can have one of three values.

**Example:**
```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

**Solution:**
```java
class Solution {
    public int orangesRotting(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int rows = grid.length, cols = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int fresh = 0;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j});
                } else if (grid[i][j] == 1) {
                    fresh++;
                }
            }
        }
        
        if (fresh == 0) return 0;
        
        int minutes = 0;
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!queue.isEmpty() && fresh > 0) {
            int size = queue.size();
            minutes++;
            
            for (int i = 0; i < size; i++) {
                int[] current = queue.poll();
                
                for (int[] dir : directions) {
                    int newRow = current[0] + dir[0];
                    int newCol = current[1] + dir[1];
                    
                    if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols && 
                        grid[newRow][newCol] == 1) {
                        grid[newRow][newCol] = 2;
                        queue.offer(new int[]{newRow, newCol});
                        fresh--;
                    }
                }
            }
        }
        
        return fresh == 0 ? minutes - 1 : -1;
    }
}
```

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n)

---

## 22. Course Schedule (Medium)

**Problem:** There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`.

**Example:**
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] prerequisite : prerequisites) {
            graph.get(prerequisite[1]).add(prerequisite[0]);
        }
        
        boolean[] visited = new boolean[numCourses];
        boolean[] recStack = new boolean[numCourses];
        
        for (int i = 0; i < numCourses; i++) {
            if (!visited[i] && hasCycle(i, graph, visited, recStack)) {
                return false;
            }
        }
        
        return true;
    }
    
    private boolean hasCycle(int node, List<List<Integer>> graph, boolean[] visited, boolean[] recStack) {
        visited[node] = true;
        recStack[node] = true;
        
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor] && hasCycle(neighbor, graph, visited, recStack)) {
                return true;
            } else if (recStack[neighbor]) {
                return true;
            }
        }
        
        recStack[node] = false;
        return false;
    }
}
```

**Time Complexity:** O(V + E)
**Space Complexity:** O(V + E)

---

## 23. Merge K Sorted Lists (Hard)

**Problem:** You are given an array of `k` linked-lists lists, each linked-list is sorted in ascending order.

**Example:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

**Solution:**
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            current.next = node;
            current = current.next;
            
            if (node.next != null) {
                pq.offer(node.next);
            }
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

---

## 24. Reverse Linked List (Easy)

**Problem:** Given the `head` of a singly linked list, reverse the list, and return the reversed list.

**Example:**
```
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
```

**Solution:**
```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 25. Maximum Subarray (Medium)

**Problem:** Given an integer array `nums`, find the contiguous subarray with the largest sum, and return its sum.

**Example:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
```

**Solution:**
```java
class Solution {
    public int maxSubArray(int[] nums) {
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

## 26. Search a 2D Matrix (Medium)

**Problem:** Write an efficient algorithm that searches for a value `target` in an `m x n` integer matrix `matrix`.

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
        
        int rows = matrix.length, cols = matrix[0].length;
        int left = 0, right = rows * cols - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int row = mid / cols;
            int col = mid % cols;
            
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

## 27. Minimum Window Substring (Hard)

**Problem:** Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window.

**Example:**
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

**Solution:**
```java
class Solution {
    public String minWindow(String s, String t) {
        if (s.length() < t.length()) return "";
        
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> have = new HashMap<>();
        
        for (char c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }
        
        int left = 0, right = 0;
        int minLen = Integer.MAX_VALUE;
        int minStart = 0;
        int required = need.size();
        int formed = 0;
        
        while (right < s.length()) {
            char c = s.charAt(right);
            have.put(c, have.getOrDefault(c, 0) + 1);
            
            if (need.containsKey(c) && have.get(c).equals(need.get(c))) {
                formed++;
            }
            
            while (left <= right && formed == required) {
                char leftChar = s.charAt(left);
                
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }
                
                have.put(leftChar, have.get(leftChar) - 1);
                if (need.containsKey(leftChar) && have.get(leftChar) < need.get(leftChar)) {
                    formed--;
                }
                left++;
            }
            right++;
        }
        
        return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(k)

---

## 28. Candy (Hard)

**Problem:** There are `n` children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

**Example:**
```
Input: ratings = [1,0,2]
Output: 5
```

**Solution:**
```java
class Solution {
    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] candies = new int[n];
        Arrays.fill(candies, 1);
        
        // Left to right pass
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }
        
        // Right to left pass
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }
        }
        
        int total = 0;
        for (int candy : candies) {
            total += candy;
        }
        
        return total;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 29. Coin Change (Medium)

**Problem:** You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

**Example:**
```
Input: coins = [1,2,5], amount = 11
Output: 3
```

**Solution:**
```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount];
    }
}
```

**Time Complexity:** O(amount × coins.length)
**Space Complexity:** O(amount)

---

## 30. Longest Repeating Character Replacement (Medium)

**Problem:** You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character.

**Example:**
```
Input: s = "ABAB", k = 1
Output: 4
```

**Solution:**
```java
class Solution {
    public int characterReplacement(String s, int k) {
        int[] count = new int[26];
        int left = 0, maxCount = 0, maxLength = 0;
        
        for (int right = 0; right < s.length(); right++) {
            count[s.charAt(right) - 'A']++;
            maxCount = Math.max(maxCount, count[s.charAt(right) - 'A']);
            
            if (right - left + 1 - maxCount > k) {
                count[s.charAt(left) - 'A']--;
                left++;
            }
            
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 31. Concatenated Words (Hard)

**Problem:** Given an array of strings `words` (without duplicates), return all the concatenated words in the given list of `words`.

**Example:**
```
Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
```

**Solution:**
```java
class Solution {
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Set<String> wordSet = new HashSet<>(Arrays.asList(words));
        List<String> result = new ArrayList<>();
        
        for (String word : words) {
            if (canForm(word, wordSet, new HashMap<>())) {
                result.add(word);
            }
        }
        
        return result;
    }
    
    private boolean canForm(String word, Set<String> wordSet, Map<String, Boolean> memo) {
        if (memo.containsKey(word)) return memo.get(word);
        
        for (int i = 1; i < word.length(); i++) {
            String left = word.substring(0, i);
            String right = word.substring(i);
            
            if (wordSet.contains(left) && (wordSet.contains(right) || canForm(right, wordSet, memo))) {
                memo.put(word, true);
                return true;
            }
        }
        
        memo.put(word, false);
        return false;
    }
}
```

**Time Complexity:** O(n × L²)
**Space Complexity:** O(n × L)

---

## 32. Climbing Stairs (Easy)

**Problem:** You are climbing a staircase. It takes `n` steps to reach the top. Each time you can either climb 1 or 2 steps.

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
        
        int prev1 = 1, prev2 = 2;
        for (int i = 3; i <= n; i++) {
            int current = prev1 + prev2;
            prev1 = prev2;
            prev2 = current;
        }
        
        return prev2;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 33. Product of Array Except Self (Medium)

**Problem:** Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

**Example:**
```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
```

**Solution:**
```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        
        // Left pass
        int leftProduct = 1;
        for (int i = 0; i < n; i++) {
            result[i] = leftProduct;
            leftProduct *= nums[i];
        }
        
        // Right pass
        int rightProduct = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] *= rightProduct;
            rightProduct *= nums[i];
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 34. Koko Eating Bananas (Medium)

**Problem:** Koko loves to eat bananas. There are `n` piles of bananas, the `i`th pile has `piles[i]` bananas.

**Example:**
```
Input: piles = [3,6,7,11], h = 8
Output: 4
```

**Solution:**
```java
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1, right = 1;
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
    
    private boolean canEatAll(int[] piles, int h, int speed) {
        int hours = 0;
        for (int pile : piles) {
            hours += (pile + speed - 1) / speed;
        }
        return hours <= h;
    }
}
```

**Time Complexity:** O(n log max(piles))
**Space Complexity:** O(1)

---

## 35. Valid Parentheses (Easy)

**Problem:** Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

**Example:**
```
Input: s = "()"
Output: true
```

**Solution:**
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) return false;
                
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 36. Generate Parentheses (Medium)

**Problem:** Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

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

## 37. Next Permutation (Medium)

**Problem:** A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

**Example:**
```
Input: nums = [1,2,3]
Output: [1,3,2]
```

**Solution:**
```java
class Solution {
    public void nextPermutation(int[] nums) {
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

## 38. Word Search (Medium)

**Problem:** Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.

**Example:**
```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
```

**Solution:**
```java
class Solution {
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, word, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean dfs(char[][] board, String word, int i, int j, int index) {
        if (index == word.length()) return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
            board[i][j] != word.charAt(index)) return false;
        
        char temp = board[i][j];
        board[i][j] = '#';
        
        boolean result = dfs(board, word, i + 1, j, index + 1) ||
                        dfs(board, word, i - 1, j, index + 1) ||
                        dfs(board, word, i, j + 1, index + 1) ||
                        dfs(board, word, i, j - 1, index + 1);
        
        board[i][j] = temp;
        return result;
    }
}
```

**Time Complexity:** O(m × n × 4^L)
**Space Complexity:** O(L)

---

## 39. Best Time to Buy and Sell Stock II (Medium)

**Problem:** You are given an integer array `prices` where `prices[i]` is the price of a given stock on the `i`th day.

**Example:**
```
Input: prices = [7,1,5,3,6,4]
Output: 7
```

**Solution:**
```java
class Solution {
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                maxProfit += prices[i] - prices[i - 1];
            }
        }
        
        return maxProfit;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 40. Min Stack (Medium)

**Problem:** Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

**Example:**
```
Input: ["MinStack","push","push","push","getMin","pop","top","getMin"]
       [[],[-2],[0],[-3],[],[],[],[]]
Output: [null,null,null,null,-3,null,0,-2]
```

**Solution:**
```java
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> minStack;
    
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int val) {
        stack.push(val);
        if (minStack.isEmpty() || val <= minStack.peek()) {
            minStack.push(val);
        }
    }
    
    public void pop() {
        if (stack.pop().equals(minStack.peek())) {
            minStack.pop();
        }
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return minStack.peek();
    }
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(n)

---

## 41. Lowest Common Ancestor of a Binary Tree (Medium)

**Problem:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

**Example:**
```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
```

**Solution:**
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        if (left != null && right != null) {
            return root;
        }
        
        return left != null ? left : right;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 42. Meeting Rooms II (Medium)

**Problem:** Given an array of meeting time intervals `intervals` where `intervals[i] = [starti, endi]`, return the minimum number of conference rooms required.

**Example:**
```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
```

**Solution:**
```java
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(intervals[0][1]);
        
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= pq.peek()) {
                pq.poll();
            }
            pq.offer(intervals[i][1]);
        }
        
        return pq.size();
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

## 43. Subarray Sum Equals K (Medium)

**Problem:** Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals to `k`.

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
            if (map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        
        return count;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 44. Flood Fill (Easy)

**Problem:** An image is represented by an `m x n` integer grid `image` where `image[i][j]` represents the pixel value of the image.

**Example:**
```
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
```

**Solution:**
```java
class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
        if (image[sr][sc] == color) return image;
        
        dfs(image, sr, sc, image[sr][sc], color);
        return image;
    }
    
    private void dfs(int[][] image, int i, int j, int oldColor, int newColor) {
        if (i < 0 || i >= image.length || j < 0 || j >= image[0].length || 
            image[i][j] != oldColor) {
            return;
        }
        
        image[i][j] = newColor;
        dfs(image, i + 1, j, oldColor, newColor);
        dfs(image, i - 1, j, oldColor, newColor);
        dfs(image, i, j + 1, oldColor, newColor);
        dfs(image, i, j - 1, oldColor, newColor);
    }
}
```

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n)

---

## 45. Number of Connected Components In An Undirected Graph (Medium)

**Problem:** You have a graph of `n` nodes. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an edge between nodes `ai` and `bi` in the graph.

**Example:**
```
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2
```

**Solution:**
```java
class Solution {
    public int countComponents(int n, int[][] edges) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        
        boolean[] visited = new boolean[n];
        int components = 0;
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs(i, graph, visited);
                components++;
            }
        }
        
        return components;
    }
    
    private void dfs(int node, List<List<Integer>> graph, boolean[] visited) {
        visited[node] = true;
        
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                dfs(neighbor, graph, visited);
            }
        }
    }
}
```

**Time Complexity:** O(V + E)
**Space Complexity:** O(V + E)

---

## 46. Integer to Roman (Medium)

**Problem:** Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

**Example:**
```
Input: num = 3
Output: "III"
```

**Solution:**
```java
class Solution {
    public String intToRoman(int num) {
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] symbols = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        
        StringBuilder result = new StringBuilder();
        
        for (int i = 0; i < values.length; i++) {
            while (num >= values[i]) {
                result.append(symbols[i]);
                num -= values[i];
            }
        }
        
        return result.toString();
    }
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

## 47. 3Sum (Medium)

**Problem:** Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

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
            if (i > 0 && nums[i] == nums[i-1]) continue;
            
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                
                if (sum == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;
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

## 48. Search In Rotated Sorted Array (Medium)

**Problem:** There is an integer array `nums` sorted in ascending order (with distinct values).

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
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
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

---

## 49. Permutations (Medium)

**Problem:** Given an array `nums` of distinct integers, return all the possible permutations. You can return the answer in any order.

**Example:**
```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Solution:**
```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> temp, int[] nums) {
        if (temp.size() == nums.length) {
            result.add(new ArrayList<>(temp));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (temp.contains(nums[i])) continue;
            temp.add(nums[i]);
            backtrack(result, temp, nums);
            temp.remove(temp.size() - 1);
        }
    }
}
```

**Time Complexity:** O(n!)
**Space Complexity:** O(n!)

---

## 50. Group Anagrams (Medium)

**Problem:** Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.

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

## 51. Unique Paths (Medium)

**Problem:** There is a robot on an `m x n` grid. The robot is initially located at the top-left corner (i.e., `grid[0][0]`).

**Example:**
```
Input: m = 3, n = 7
Output: 28
```

**Solution:**
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        
        return dp[m-1][n-1];
    }
}
```

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n)

---

## 52. Edit Distance (Hard)

**Problem:** Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`.

**Example:**
```
Input: word1 = "horse", word2 = "ros"
Output: 3
```

**Solution:**
```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        
        return dp[m][n];
    }
}
```

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n)

---

## 53. Binary Tree Right Side View (Medium)

**Problem:** Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

**Example:**
```
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]
```

**Solution:**
```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (i == size - 1) {
                    result.add(node.val);
                }
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(w)

---

## 54. Course Schedule II (Medium)

**Problem:** There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`.

**Example:**
```
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3]
```

**Solution:**
```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] prerequisite : prerequisites) {
            graph.get(prerequisite[1]).add(prerequisite[0]);
        }
        
        boolean[] visited = new boolean[numCourses];
        boolean[] recStack = new boolean[numCourses];
        List<Integer> result = new ArrayList<>();
        
        for (int i = 0; i < numCourses; i++) {
            if (!visited[i] && hasCycle(i, graph, visited, recStack, result)) {
                return new int[0];
            }
        }
        
        Collections.reverse(result);
        return result.stream().mapToInt(Integer::intValue).toArray();
    }
    
    private boolean hasCycle(int node, List<List<Integer>> graph, boolean[] visited, 
                           boolean[] recStack, List<Integer> result) {
        visited[node] = true;
        recStack[node] = true;
        
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor] && hasCycle(neighbor, graph, visited, recStack, result)) {
                return true;
            } else if (recStack[neighbor]) {
                return true;
            }
        }
        
        recStack[node] = false;
        result.add(node);
        return false;
    }
}
```

**Time Complexity:** O(V + E)
**Space Complexity:** O(V + E)

---

## 55. Kth Largest Element In An Array (Medium)

**Problem:** Find the kth largest element in an unsorted array.

**Example:**
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

**Solution:**
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        
        for (int num : nums) {
            pq.offer(num);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        return pq.peek();
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

---

## 56. Missing Number (Easy)

**Problem:** Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return the only number in the range that is missing from the array.

**Example:**
```
Input: nums = [3,0,1]
Output: 2
```

**Solution:**
```java
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int expectedSum = n * (n + 1) / 2;
        int actualSum = 0;
        
        for (int num : nums) {
            actualSum += num;
        }
        
        return expectedSum - actualSum;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 57. LFU Cache (Hard)

**Problem:** Design and implement a data structure for a Least Frequently Used (LFU) cache.

**Example:**
```
Input: ["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
       [[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
Output: [null, null, null, 1, null, -1, 3, null, -1, 3, 4]
```

**Solution:**
```java
class LFUCache {
    private Map<Integer, Node> cache;
    private Map<Integer, LinkedHashSet<Node>> frequencyMap;
    private int capacity, minFrequency;
    
    public LFUCache(int capacity) {
        this.capacity = capacity;
        cache = new HashMap<>();
        frequencyMap = new HashMap<>();
        minFrequency = 0;
    }
    
    public int get(int key) {
        if (!cache.containsKey(key)) return -1;
        
        Node node = cache.get(key);
        updateFrequency(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        if (capacity == 0) return;
        
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            node.value = value;
            updateFrequency(node);
        } else {
            if (cache.size() >= capacity) {
                LinkedHashSet<Node> minFreqSet = frequencyMap.get(minFrequency);
                Node lfu = minFreqSet.iterator().next();
                minFreqSet.remove(lfu);
                cache.remove(lfu.key);
            }
            
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            frequencyMap.computeIfAbsent(1, k -> new LinkedHashSet<>()).add(newNode);
            minFrequency = 1;
        }
    }
    
    private void updateFrequency(Node node) {
        int freq = node.frequency;
        frequencyMap.get(freq).remove(node);
        
        if (freq == minFrequency && frequencyMap.get(freq).isEmpty()) {
            minFrequency++;
        }
        
        node.frequency++;
        frequencyMap.computeIfAbsent(node.frequency, k -> new LinkedHashSet<>()).add(node);
    }
    
    class Node {
        int key, value, frequency;
        
        Node(int key, int value) {
            this.key = key;
            this.value = value;
            this.frequency = 1;
        }
    }
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(capacity)

---

## 58. Capacity to Ship Packages Within D Days (Medium)

**Problem:** A conveyor belt has packages that must be shipped from one port to another within `days` days.

**Example:**
```
Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
```

**Solution:**
```java
class Solution {
    public int shipWithinDays(int[] weights, int days) {
        int left = 0, right = 0;
        for (int weight : weights) {
            left = Math.max(left, weight);
            right += weight;
        }
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (canShip(weights, days, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return left;
    }
    
    private boolean canShip(int[] weights, int days, int capacity) {
        int currentWeight = 0;
        int daysNeeded = 1;
        
        for (int weight : weights) {
            if (currentWeight + weight > capacity) {
                daysNeeded++;
                currentWeight = weight;
            } else {
                currentWeight += weight;
            }
        }
        
        return daysNeeded <= days;
    }
}
```

**Time Complexity:** O(n log sum(weights))
**Space Complexity:** O(1)

---

## 59. Reverse Integer (Medium)

**Problem:** Given a signed 32-bit integer `x`, return `x` with its digits reversed.

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

## 60. Roman to Integer (Easy)

**Problem:** Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

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

## 61. Reverse Nodes In K Group (Hard)

**Problem:** Given the `head` of a linked list, reverse the nodes of the list `k` at a time, and return the modified list.

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

## 62. Valid Sudoku (Medium)

**Problem:** Determine if a `9 x 9` Sudoku board is valid.

**Example:**
```
Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            if (!isValidRow(board, i) || !isValidColumn(board, i)) {
                return false;
            }
        }
        
        for (int i = 0; i < 9; i += 3) {
            for (int j = 0; j < 9; j += 3) {
                if (!isValidBox(board, i, j)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    private boolean isValidRow(char[][] board, int row) {
        Set<Character> set = new HashSet<>();
        for (int j = 0; j < 9; j++) {
            char c = board[row][j];
            if (c != '.' && !set.add(c)) {
                return false;
            }
        }
        return true;
    }
    
    private boolean isValidColumn(char[][] board, int col) {
        Set<Character> set = new HashSet<>();
        for (int i = 0; i < 9; i++) {
            char c = board[i][col];
            if (c != '.' && !set.add(c)) {
                return false;
            }
        }
        return true;
    }
    
    private boolean isValidBox(char[][] board, int startRow, int startCol) {
        Set<Character> set = new HashSet<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                char c = board[startRow + i][startCol + j];
                if (c != '.' && !set.add(c)) {
                    return false;
                }
            }
        }
        return true;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(n)

---

## 63. Combination Sum (Medium)

**Problem:** Given an array of distinct integers `candidates` and a target integer `target`, return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`.

**Example:**
```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
```

**Solution:**
```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), candidates, target, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> temp, 
                          int[] candidates, int target, int start) {
        if (target == 0) {
            result.add(new ArrayList<>(temp));
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            if (candidates[i] <= target) {
                temp.add(candidates[i]);
                backtrack(result, temp, candidates, target - candidates[i], i);
                temp.remove(temp.size() - 1);
            }
        }
    }
}
```

**Time Complexity:** O(n^(target/min))
**Space Complexity:** O(target/min)

---

## 64. Rotate Image (Medium)

**Problem:** You are given an `n x n` 2D `matrix` representing an image, rotate the image by 90 degrees (clockwise).

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

## 65. N Queens (Hard)

**Problem:** The n-queens puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

**Example:**
```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
```

**Solution:**
```java
class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        char[][] board = new char[n][n];
        for (char[] row : board) {
            Arrays.fill(row, '.');
        }
        
        backtrack(result, board, 0, n);
        return result;
    }
    
    private void backtrack(List<List<String>> result, char[][] board, int row, int n) {
        if (row == n) {
            result.add(constructBoard(board));
            return;
        }
        
        for (int col = 0; col < n; col++) {
            if (isValid(board, row, col, n)) {
                board[row][col] = 'Q';
                backtrack(result, board, row + 1, n);
                board[row][col] = '.';
            }
        }
    }
    
    private boolean isValid(char[][] board, int row, int col, int n) {
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') return false;
        }
        
        // Check diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') return false;
        }
        
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') return false;
        }
        
        return true;
    }
    
    private List<String> constructBoard(char[][] board) {
        List<String> result = new ArrayList<>();
        for (char[] row : board) {
            result.add(new String(row));
        }
        return result;
    }
}
```

**Time Complexity:** O(n!)
**Space Complexity:** O(n²)

---

## 66. Merge Sorted Array (Easy)

**Problem:** You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively.

**Example:**
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```

**Solution:**
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1, p = m + n - 1;
        
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

## 67. Copy List With Random Pointer (Medium)

**Problem:** A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

**Example:**
```
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**Solution:**
```java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        
        // Step 1: Create copy nodes and insert them after original nodes
        Node current = head;
        while (current != null) {
            Node copy = new Node(current.val);
            copy.next = current.next;
            current.next = copy;
            current = copy.next;
        }
        
        // Step 2: Set random pointers for copy nodes
        current = head;
        while (current != null) {
            if (current.random != null) {
                current.next.random = current.random.next;
            }
            current = current.next.next;
        }
        
        // Step 3: Separate original and copy lists
        Node dummy = new Node(0);
        Node copyCurrent = dummy;
        current = head;
        
        while (current != null) {
            copyCurrent.next = current.next;
            copyCurrent = copyCurrent.next;
            current.next = current.next.next;
            current = current.next;
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## Summary

These Amazon tagged problems cover essential algorithmic concepts including:

- **Arrays & Hashing:** Two Sum, Top K Frequent Elements, Subarray Sum Equals K
- **Two Pointers:** Longest Substring Without Repeating Characters
- **Sliding Window:** Longest Repeating Character Replacement
- **Stack:** Valid Parentheses, Min Stack
- **Linked Lists:** Add Two Numbers, Merge K Sorted Lists, Reverse Linked List, Reverse Nodes In K Group, Copy List With Random Pointer
- **Dynamic Programming:** House Robber, Maximum Subarray, Climbing Stairs, Edit Distance, Unique Paths, Coin Change
- **Backtracking:** Generate Parentheses, Word Search, Permutations, Combination Sum, N Queens
- **Graph:** Number of Islands, Course Schedule, Course Schedule II, Number of Connected Components In An Undirected Graph, Flood Fill
- **Design:** LRU Cache, LFU Cache
- **String:** Reorganize String, Longest Palindromic Substring, Concatenated Words, Valid Sudoku, Group Anagrams
- **Greedy:** Candy, Best Time to Buy And Sell Stock, Best Time to Buy And Sell Stock II
- **Binary Search:** Find First And Last Position of Element In Sorted Array, Search a 2D Matrix, Search In Rotated Sorted Array, Koko Eating Bananas, Capacity to Ship Packages Within D Days
- **Tree:** Lowest Common Ancestor of a Binary Tree, Binary Tree Right Side View
- **Math:** Reverse Integer, Integer to Roman, Roman to Integer, Missing Number
- **Matrix:** Rotate Image, Search a 2D Matrix
- **Sorting:** Merge Sorted Array
- **Interval:** Meeting Rooms II
- **Heap:** Kth Largest Element In An Array

Each solution includes detailed explanations, time and space complexity analysis, and follows best practices for coding interviews. These problems are commonly asked in Amazon technical interviews and represent the core algorithmic concepts that candidates should master.
