---
title: 1-D Dynamic Programming
description: Java solutions with explanations, time and space complexity for 1D Dynamic Programming problems from Blind 75.
date: "June 1 2025"
order: 13
---

# 1-D Dynamic Programming

This section covers problems that can be solved using 1-dimensional dynamic programming.

## 1. Climbing Stairs (Easy)

**Problem:** You are climbing a staircase. It takes `n` steps to reach the top. Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

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

## 2. House Robber (Medium)

**Problem:** You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

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

## 3. House Robber II (Medium)

**Problem:** You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle.

**Example:**
```
Input: nums = [2,3,2]
Output: 3
```

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
        if (start > end) return 0;
        
        int prev1 = 0, prev2 = nums[start];
        
        for (int i = start + 1; i <= end; i++) {
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

## 4. Longest Palindromic Substring (Medium)

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
        if (s.length() < 2) return s;
        
        int start = 0, maxLength = 1;
        
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            
            int len = Math.max(len1, len2);
            if (len > maxLength) {
                start = i - (len - 1) / 2;
                maxLength = len;
            }
        }
        
        return s.substring(start, start + maxLength);
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

## 5. Palindromic Substrings (Medium)

**Problem:** Given a string `s`, return the number of palindromic substrings in it.

**Example:**
```
Input: s = "aaa"
Output: 6
```

**Solution:**
```java
class Solution {
    public int countSubstrings(String s) {
        int count = 0;
        
        for (int i = 0; i < s.length(); i++) {
            count += expandAroundCenter(s, i, i);
            count += expandAroundCenter(s, i, i + 1);
        }
        
        return count;
    }
    
    private int expandAroundCenter(String s, int left, int right) {
        int count = 0;
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            count++;
            left--;
            right++;
        }
        return count;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)

---

## 6. Decode Ways (Medium)

**Problem:** A message containing letters from `A-Z` can be encoded into numbers using the following mapping.

**Example:**
```
Input: s = "12"
Output: 2
```

**Solution:**
```java
class Solution {
    public int numDecodings(String s) {
        if (s.length() == 0) return 0;
        
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        
        for (int i = 2; i <= s.length(); i++) {
            int oneDigit = Integer.parseInt(s.substring(i - 1, i));
            int twoDigits = Integer.parseInt(s.substring(i - 2, i));
            
            if (oneDigit >= 1) {
                dp[i] += dp[i - 1];
            }
            
            if (twoDigits >= 10 && twoDigits <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        
        return dp[s.length()];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 7. Coin Change (Medium)

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

**Time Complexity:** O(amount * coins.length)
**Space Complexity:** O(amount)

---

## 8. Maximum Product Subarray (Medium)

**Problem:** Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

**Example:**
```
Input: nums = [2,3,-2,4]
Output: 6
```

**Solution:**
```java
class Solution {
    public int maxProduct(int[] nums) {
        if (nums.length == 0) return 0;
        
        int maxSoFar = nums[0];
        int maxEndingHere = nums[0];
        int minEndingHere = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            int temp = maxEndingHere;
            maxEndingHere = Math.max(nums[i], Math.max(maxEndingHere * nums[i], minEndingHere * nums[i]));
            minEndingHere = Math.min(nums[i], Math.min(temp * nums[i], minEndingHere * nums[i]));
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 9. Word Break (Medium)

**Problem:** Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

**Example:**
```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[s.length()];
    }
}
```

**Time Complexity:** O(n³)
**Space Complexity:** O(n)

---

## 10. Longest Increasing Subsequence (Medium)

**Problem:** Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

**Example:**
```
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
```

**Solution:**
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        
        int maxLength = 1;
        
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxLength = Math.max(maxLength, dp[i]);
        }
        
        return maxLength;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(n)

## Key Takeaways

1. **State Definition**: Clearly define what dp[i] represents
2. **Base Cases**: Handle edge cases properly
3. **State Transition**: Understand the relationship between states
4. **Optimization**: Look for ways to reduce space complexity
5. **Subproblems**: Break down complex problems into smaller subproblems 