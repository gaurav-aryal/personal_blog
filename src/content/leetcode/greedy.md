---
title: Greedy
description: Java solutions with explanations, time and space complexity for Greedy problems.
date: "June 1 2025"
---

# Greedy Pattern

Greedy algorithms make locally optimal choices at each step with the hope of finding a global optimum. They're particularly useful for:
- Optimization problems
- Scheduling problems
- Resource allocation
- Path finding
- Interval problems

## 1. Maximum Subarray (Medium)

**Problem:** Given an integer array nums, find the contiguous subarray with the largest sum and return its sum.

**Solution:**
```java
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSoFar = nums[0];
        int currentMax = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            // Either extend the current subarray or start a new one
            currentMax = Math.max(nums[i], currentMax + nums[i]);
            maxSoFar = Math.max(maxSoFar, currentMax);
        }
        
        return maxSoFar;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 2. Jump Game (Medium)

**Problem:** You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position. Return true if you can reach the last index, or false otherwise.

**Solution:**
```java
class Solution {
    public boolean canJump(int[] nums) {
        int maxReach = 0;
        
        for (int i = 0; i < nums.length; i++) {
            if (i > maxReach) return false;
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= nums.length - 1) return true;
        }
        
        return true;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 3. Jump Game II (Medium)

**Problem:** Given an array of non-negative integers nums, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Your goal is to reach the last index in the minimum number of jumps.

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

## 4. Gas Station (Medium)

**Problem:** There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]. You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station.

**Solution:**
```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int totalGas = 0;
        int currentGas = 0;
        int start = 0;
        
        for (int i = 0; i < gas.length; i++) {
            totalGas += gas[i] - cost[i];
            currentGas += gas[i] - cost[i];
            
            if (currentGas < 0) {
                start = i + 1;
                currentGas = 0;
            }
        }
        
        return totalGas >= 0 ? start : -1;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 5. Hand of Straights (Medium)

**Problem:** Alice has a hand of cards, given as an array of integers. Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.

**Solution:**
```java
class Solution {
    public boolean isNStraightHand(int[] hand, int W) {
        if (hand.length % W != 0) return false;
        
        TreeMap<Integer, Integer> count = new TreeMap<>();
        for (int card : hand) {
            count.put(card, count.getOrDefault(card, 0) + 1);
        }
        
        while (!count.isEmpty()) {
            int first = count.firstKey();
            for (int i = 0; i < W; i++) {
                int card = first + i;
                if (!count.containsKey(card)) return false;
                
                count.put(card, count.get(card) - 1);
                if (count.get(card) == 0) count.remove(card);
            }
        }
        
        return true;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

## 6. Merge Triplets to Form Target Triplet (Medium)

**Problem:** A triplet is an array of three integers. You are given a 2D integer array triplets, where triplets[i] = [ai, bi, ci] describes the ith triplet. You are also given an integer array target = [x, y, z] that describes the triplet you want to obtain.

**Solution:**
```java
class Solution {
    public boolean mergeTriplets(int[][] triplets, int[] target) {
        boolean[] found = new boolean[3];
        
        for (int[] triplet : triplets) {
            if (triplet[0] > target[0] || triplet[1] > target[1] || triplet[2] > target[2]) {
                continue;
            }
            
            for (int i = 0; i < 3; i++) {
                if (triplet[i] == target[i]) {
                    found[i] = true;
                }
            }
        }
        
        return found[0] && found[1] && found[2];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 7. Partition Labels (Medium)

**Problem:** You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

**Solution:**
```java
class Solution {
    public List<Integer> partitionLabels(String s) {
        int[] last = new int[26];
        for (int i = 0; i < s.length(); i++) {
            last[s.charAt(i) - 'a'] = i;
        }
        
        List<Integer> result = new ArrayList<>();
        int start = 0, end = 0;
        
        for (int i = 0; i < s.length(); i++) {
            end = Math.max(end, last[s.charAt(i) - 'a']);
            if (i == end) {
                result.add(end - start + 1);
                start = end + 1;
            }
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 8. Valid Parenthesis String (Medium)

**Problem:** Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.

**Solution:**
```java
class Solution {
    public boolean checkValidString(String s) {
        int minOpen = 0, maxOpen = 0;
        
        for (char c : s.toCharArray()) {
            if (c == '(') {
                minOpen++;
                maxOpen++;
            } else if (c == ')') {
                minOpen--;
                maxOpen--;
            } else {
                minOpen--;
                maxOpen++;
            }
            
            if (maxOpen < 0) return false;
            minOpen = Math.max(0, minOpen);
        }
        
        return minOpen == 0;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## Key Takeaways

1. Greedy algorithms are perfect for:
   - Optimization problems
   - Scheduling problems
   - Resource allocation
   - Path finding
   - Interval problems

2. Common patterns:
   - Local optimal choices
   - Sorting and processing
   - Two-pointer technique
   - Interval merging
   - Resource allocation

3. Tips:
   - Prove the greedy choice is optimal
   - Consider edge cases
   - Look for sorting opportunities
   - Think about local vs global optimality
   - Consider using data structures for efficiency 