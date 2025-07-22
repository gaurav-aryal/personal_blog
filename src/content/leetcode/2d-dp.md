---
title: 2D Dynamic Programming
description: Java solutions with explanations, time and space complexity for 2D Dynamic Programming problems.
date: "June 1 2025"
---

# 2-D Dynamic Programming Pattern

2-D Dynamic Programming extends the concept of 1-D DP to solve problems that require tracking multiple states or dimensions. It's particularly useful for:
- Grid-based problems
- String matching and comparison
- Matrix operations
- Path finding in 2D space
- Sequence alignment

## 1. Unique Paths (Medium)

**Problem:** There is a robot on an m x n grid. The robot is initially located at the top-left corner and tries to move to the bottom-right corner. The robot can only move either down or right at any point in time.

**Solution:**
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        
        // Initialize first row and column
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int j = 0; j < n; j++) dp[0][j] = 1;
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        
        return dp[m - 1][n - 1];
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## 2. Longest Common Subsequence (Medium)

**Problem:** Given two strings text1 and text2, return the length of their longest common subsequence.

**Solution:**
```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## 3. Best Time to Buy and Sell Stock with Cooldown (Medium)

**Problem:** You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit you can achieve with a cooldown period of one day.

**Solution:**
```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) return 0;
        
        int n = prices.length;
        int[][] dp = new int[n][2];
        
        // dp[i][0] = max profit on day i with no stock
        // dp[i][1] = max profit on day i with stock
        
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[1][0] = Math.max(dp[0][0], dp[0][1] + prices[1]);
        dp[1][1] = Math.max(dp[0][1], -prices[1]);
        
        for (int i = 2; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i]);
        }
        
        return dp[n - 1][0];
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

## 4. Coin Change II (Medium)

**Problem:** You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the number of combinations that make up that amount.

**Solution:**
```java
class Solution {
    public int change(int amount, int[] coins) {
        int[][] dp = new int[coins.length + 1][amount + 1];
        dp[0][0] = 1;
        
        for (int i = 1; i <= coins.length; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= amount; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= coins[i - 1]) {
                    dp[i][j] += dp[i][j - coins[i - 1]];
                }
            }
        }
        
        return dp[coins.length][amount];
    }
}
```

**Time Complexity:** O(amount * coins.length)
**Space Complexity:** O(amount * coins.length)

## 5. Target Sum (Medium)

**Problem:** You are given an integer array nums and an integer target. You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums.

**Solution:**
```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) sum += num;
        
        if (Math.abs(target) > sum) return 0;
        
        int[][] dp = new int[nums.length + 1][2 * sum + 1];
        dp[0][sum] = 1;
        
        for (int i = 1; i <= nums.length; i++) {
            for (int j = 0; j <= 2 * sum; j++) {
                if (j - nums[i - 1] >= 0) {
                    dp[i][j] += dp[i - 1][j - nums[i - 1]];
                }
                if (j + nums[i - 1] <= 2 * sum) {
                    dp[i][j] += dp[i - 1][j + nums[i - 1]];
                }
            }
        }
        
        return dp[nums.length][sum + target];
    }
}
```

**Time Complexity:** O(n * sum)
**Space Complexity:** O(n * sum)

## 6. Interleaving String (Medium)

**Problem:** Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

**Solution:**
```java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.length() + s2.length() != s3.length()) return false;
        
        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
        dp[0][0] = true;
        
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i > 0 && s1.charAt(i - 1) == s3.charAt(i + j - 1)) {
                    dp[i][j] |= dp[i - 1][j];
                }
                if (j > 0 && s2.charAt(j - 1) == s3.charAt(i + j - 1)) {
                    dp[i][j] |= dp[i][j - 1];
                }
            }
        }
        
        return dp[s1.length()][s2.length()];
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## 7. Longest Increasing Path in a Matrix (Hard)

**Problem:** Given an m x n integers matrix, return the length of the longest increasing path in matrix.

**Solution:**
```java
class Solution {
    private int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    
    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0) return 0;
        
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        int maxLen = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxLen = Math.max(maxLen, dfs(matrix, i, j, dp));
            }
        }
        
        return maxLen;
    }
    
    private int dfs(int[][] matrix, int i, int j, int[][] dp) {
        if (dp[i][j] != 0) return dp[i][j];
        
        int max = 1;
        for (int[] dir : directions) {
            int x = i + dir[0], y = j + dir[1];
            
            if (x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length 
                && matrix[x][y] > matrix[i][j]) {
                max = Math.max(max, 1 + dfs(matrix, x, y, dp));
            }
        }
        
        dp[i][j] = max;
        return max;
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## 8. Distinct Subsequences (Hard)

**Problem:** Given two strings s and t, return the number of distinct subsequences of s which equals t.

**Solution:**
```java
class Solution {
    public int numDistinct(String s, String t) {
        int m = s.length(), n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 0; i <= m; i++) {
            dp[i][0] = 1;
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        
        return dp[m][n];
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## 9. Edit Distance (Hard)

**Problem:** Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

**Solution:**
```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], 
                                      Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        
        return dp[m][n];
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## 10. Burst Balloons (Hard)

**Problem:** You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums. You are asked to burst all the balloons.

**Solution:**
```java
class Solution {
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] newNums = new int[n + 2];
        newNums[0] = newNums[n + 1] = 1;
        for (int i = 0; i < n; i++) {
            newNums[i + 1] = nums[i];
        }
        
        int[][] dp = new int[n + 2][n + 2];
        
        for (int len = 1; len <= n; len++) {
            for (int left = 1; left <= n - len + 1; left++) {
                int right = left + len - 1;
                for (int k = left; k <= right; k++) {
                    dp[left][right] = Math.max(dp[left][right],
                        newNums[left - 1] * newNums[k] * newNums[right + 1] +
                        dp[left][k - 1] + dp[k + 1][right]);
                }
            }
        }
        
        return dp[1][n];
    }
}
```

**Time Complexity:** O(n³)
**Space Complexity:** O(n²)

## 11. Regular Expression Matching (Hard)

**Problem:** Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*'.

**Solution:**
```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        
        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '*') {
                dp[0][j] = dp[0][j - 2];
            }
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2];
                    if (p.charAt(j - 2) == '.' || p.charAt(j - 2) == s.charAt(i - 1)) {
                        dp[i][j] |= dp[i - 1][j];
                    }
                }
            }
        }
        
        return dp[m][n];
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m * n)

## Key Takeaways

1. 2-D DP is perfect for:
   - Grid-based problems
   - String matching and comparison
   - Matrix operations
   - Path finding in 2D space
   - Sequence alignment

2. Common patterns:
   - State transition matrices
   - Multiple state tracking
   - Grid traversal
   - String comparison
   - Path counting

3. Tips:
   - Identify the two dimensions
   - Consider state transitions
   - Handle edge cases
   - Optimize space when possible
   - Think about initialization 