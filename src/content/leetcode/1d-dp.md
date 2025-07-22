---
title: 1D Dynamic Programming
description: Java solutions with explanations, time and space complexity for 1D Dynamic Programming problems.
date: "June 1 2025"
---

# 1-D Dynamic Programming Pattern

1-D Dynamic Programming is a technique that solves complex problems by breaking them down into simpler subproblems. It's particularly useful for:
- Optimization problems
- Counting problems
- Path finding
- Sequence problems
- Resource allocation

## 1. Climbing Stairs (Easy)

**Problem:** You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Solution:**
```java
class Solution {
    public int climbStairs(int n) {
        if (n <= 2) return n;
        
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        return dp[n];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

## 2. Min Cost Climbing Stairs (Easy)

**Problem:** You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

**Solution:**
```java
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        
        return dp[n];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

## 3. House Robber (Medium)

**Problem:** You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected.

**Solution:**
```java
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        
        return dp[nums.length - 1];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

## 4. House Robber II (Medium)

**Problem:** All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one.

**Solution:**
```java
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        return Math.max(robRange(nums, 0, nums.length - 2),
                       robRange(nums, 1, nums.length - 1));
    }
    
    private int robRange(int[] nums, int start, int end) {
        int prev2 = 0, prev1 = 0;
        
        for (int i = start; i <= end; i++) {
            int curr = Math.max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }
        
        return prev1;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 5. Longest Palindromic Substring (Medium)

**Problem:** Given a string s, return the longest palindromic substring in s.

**Solution:**
```java
class Solution {
    public String longestPalindrome(String s) {
        int n = s.length();
        boolean[] dp = new boolean[n];
        int start = 0, maxLen = 1;
        
        for (int i = n - 1; i >= 0; i--) {
            for (int j = n - 1; j >= i; j--) {
                dp[j] = s.charAt(i) == s.charAt(j) && 
                       (j - i < 3 || dp[j - 1]);
                
                if (dp[j] && j - i + 1 > maxLen) {
                    start = i;
                    maxLen = j - i + 1;
                }
            }
        }
        
        return s.substring(start, start + maxLen);
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(n)

## 6. Palindromic Substrings (Medium)

**Problem:** Given a string s, return the number of palindromic substrings in it.

**Solution:**
```java
class Solution {
    public int countSubstrings(String s) {
        int n = s.length();
        boolean[] dp = new boolean[n];
        int count = 0;
        
        for (int i = n - 1; i >= 0; i--) {
            for (int j = n - 1; j >= i; j--) {
                dp[j] = s.charAt(i) == s.charAt(j) && 
                       (j - i < 3 || dp[j - 1]);
                
                if (dp[j]) count++;
            }
        }
        
        return count;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(n)

## 7. Decode Ways (Medium)

**Problem:** A message containing letters from A-Z can be encoded into numbers using the following mapping: 'A' -> "1", 'B' -> "2", ..., 'Z' -> "26".

**Solution:**
```java
class Solution {
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) return 0;
        
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        
        for (int i = 2; i <= n; i++) {
            int oneDigit = Integer.parseInt(s.substring(i - 1, i));
            int twoDigits = Integer.parseInt(s.substring(i - 2, i));
            
            if (oneDigit >= 1 && oneDigit <= 9) {
                dp[i] += dp[i - 1];
            }
            
            if (twoDigits >= 10 && twoDigits <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        
        return dp[n];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

## 8. Coin Change (Medium)

**Problem:** You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

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

**Time Complexity:** O(amount * coins.length)
**Space Complexity:** O(amount)

## 9. Maximum Product Subarray (Medium)

**Problem:** Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product.

**Solution:**
```java
class Solution {
    public int maxProduct(int[] nums) {
        int maxSoFar = nums[0];
        int currMax = nums[0];
        int currMin = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            int temp = currMax;
            currMax = Math.max(Math.max(currMax * nums[i], currMin * nums[i]), nums[i]);
            currMin = Math.min(Math.min(temp * nums[i], currMin * nums[i]), nums[i]);
            maxSoFar = Math.max(maxSoFar, currMax);
        }
        
        return maxSoFar;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 10. Word Break (Medium)

**Problem:** Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

**Solution:**
```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[s.length()];
    }
}
```

**Time Complexity:** O(n³) where n is the length of s
**Space Complexity:** O(n)

## 11. Longest Increasing Subsequence (Medium)

**Problem:** Given an integer array nums, return the length of the longest strictly increasing subsequence.

**Solution:**
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int maxLen = 1;
        
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                    maxLen = Math.max(maxLen, dp[i]);
                }
            }
        }
        
        return maxLen;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(n)

## 12. Partition Equal Subset Sum (Medium)

**Problem:** Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

**Solution:**
```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        
        if (sum % 2 != 0) return false;
        
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        
        for (int num : nums) {
            for (int i = target; i >= num; i--) {
                dp[i] = dp[i] || dp[i - num];
            }
        }
        
        return dp[target];
    }
}
```

**Time Complexity:** O(n * target) where target is sum/2
**Space Complexity:** O(target)

## Key Takeaways

1. 1-D DP is perfect for:
   - Optimization problems
   - Counting problems
   - Path finding
   - Sequence problems
   - Resource allocation

2. Common patterns:
   - State transition equations
   - Base case initialization
   - Bottom-up or top-down approaches
   - Space optimization
   - Rolling arrays

3. Tips:
   - Identify the recurrence relation
   - Choose appropriate base cases
   - Consider space optimization
   - Handle edge cases carefully
   - Think about state transitions 