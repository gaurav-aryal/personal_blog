---
title: Backtracking
description: Java solutions with explanations, time and space complexity for Backtracking problems.
date: "June 1 2025"
---

# Backtracking Pattern

Backtracking is an algorithmic technique that considers searching every possible combination in order to solve a computational problem. It's particularly useful for:
- Generating all possible combinations/permutations
- Solving constraint satisfaction problems
- Finding all possible solutions
- Path finding in mazes/grids
- Game solving (like N-Queens)

## 1. Subsets (Medium)

**Problem:** Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets.

**Solution:**
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> temp, int[] nums, int start) {
        result.add(new ArrayList<>(temp));
        
        for (int i = start; i < nums.length; i++) {
            temp.add(nums[i]);
            backtrack(result, temp, nums, i + 1);
            temp.remove(temp.size() - 1);
        }
    }
}
```

**Time Complexity:** O(n * 2^n) where n is the length of nums
**Space Complexity:** O(n) for recursion stack

## 2. Combination Sum (Medium)

**Problem:** Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target.

**Solution:**
```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), candidates, target, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> temp, 
                          int[] candidates, int remain, int start) {
        if (remain < 0) return;
        if (remain == 0) {
            result.add(new ArrayList<>(temp));
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            temp.add(candidates[i]);
            backtrack(result, temp, candidates, remain - candidates[i], i);
            temp.remove(temp.size() - 1);
        }
    }
}
```

**Time Complexity:** O(n^(target/min)) where n is the length of candidates
**Space Complexity:** O(target/min) for recursion stack

## 3. Permutations (Medium)

**Problem:** Given an array nums of distinct integers, return all the possible permutations.

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

**Time Complexity:** O(n!) where n is the length of nums
**Space Complexity:** O(n) for recursion stack

## 4. Subsets II (Medium)

**Problem:** Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

**Solution:**
```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(result, new ArrayList<>(), nums, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> temp, 
                          int[] nums, int start) {
        result.add(new ArrayList<>(temp));
        
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i-1]) continue;
            temp.add(nums[i]);
            backtrack(result, temp, nums, i + 1);
            temp.remove(temp.size() - 1);
        }
    }
}
```

**Time Complexity:** O(n * 2^n) where n is the length of nums
**Space Complexity:** O(n) for recursion stack

## 5. Combination Sum II (Medium)

**Problem:** Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

**Solution:**
```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(result, new ArrayList<>(), candidates, target, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> temp, 
                          int[] candidates, int remain, int start) {
        if (remain < 0) return;
        if (remain == 0) {
            result.add(new ArrayList<>(temp));
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i-1]) continue;
            temp.add(candidates[i]);
            backtrack(result, temp, candidates, remain - candidates[i], i + 1);
            temp.remove(temp.size() - 1);
        }
    }
}
```

**Time Complexity:** O(2^n) where n is the length of candidates
**Space Complexity:** O(n) for recursion stack

## 6. Word Search (Medium)

**Problem:** Given an m x n grid of characters board and a string word, return true if word exists in the grid.

**Solution:**
```java
class Solution {
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (backtrack(board, word, 0, i, j)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean backtrack(char[][] board, String word, int index, int i, int j) {
        if (index == word.length()) return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length 
            || board[i][j] != word.charAt(index)) {
            return false;
        }
        
        char temp = board[i][j];
        board[i][j] = '#';
        
        boolean found = backtrack(board, word, index + 1, i + 1, j) ||
                       backtrack(board, word, index + 1, i - 1, j) ||
                       backtrack(board, word, index + 1, i, j + 1) ||
                       backtrack(board, word, index + 1, i, j - 1);
        
        board[i][j] = temp;
        return found;
    }
}
```

**Time Complexity:** O(m * n * 4^L) where m,n are dimensions and L is word length
**Space Complexity:** O(L) for recursion stack

## 7. Palindrome Partitioning (Medium)

**Problem:** Given a string s, partition s such that every substring of the partition is a palindrome.

**Solution:**
```java
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), s, 0);
        return result;
    }
    
    private void backtrack(List<List<String>> result, List<String> temp, 
                          String s, int start) {
        if (start == s.length()) {
            result.add(new ArrayList<>(temp));
            return;
        }
        
        for (int i = start; i < s.length(); i++) {
            if (isPalindrome(s, start, i)) {
                temp.add(s.substring(start, i + 1));
                backtrack(result, temp, s, i + 1);
                temp.remove(temp.size() - 1);
            }
        }
    }
    
    private boolean isPalindrome(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left++) != s.charAt(right--)) {
                return false;
            }
        }
        return true;
    }
}
```

**Time Complexity:** O(n * 2^n) where n is the length of s
**Space Complexity:** O(n) for recursion stack

## 8. Letter Combinations of a Phone Number (Medium)

**Problem:** Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

**Solution:**
```java
class Solution {
    private String[] mapping = {"", "", "abc", "def", "ghi", "jkl", 
                              "mno", "pqrs", "tuv", "wxyz"};
    
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if (digits.length() == 0) return result;
        backtrack(result, new StringBuilder(), digits, 0);
        return result;
    }
    
    private void backtrack(List<String> result, StringBuilder temp, 
                          String digits, int index) {
        if (index == digits.length()) {
            result.add(temp.toString());
            return;
        }
        
        String letters = mapping[digits.charAt(index) - '0'];
        for (char c : letters.toCharArray()) {
            temp.append(c);
            backtrack(result, temp, digits, index + 1);
            temp.deleteCharAt(temp.length() - 1);
        }
    }
}
```

**Time Complexity:** O(4^n) where n is the length of digits
**Space Complexity:** O(n) for recursion stack

## 9. N Queens (Hard)

**Problem:** The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

**Solution:**
```java
class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        char[][] board = new char[n][n];
        for (char[] row : board) {
            Arrays.fill(row, '.');
        }
        backtrack(result, board, 0);
        return result;
    }
    
    private void backtrack(List<List<String>> result, char[][] board, int row) {
        if (row == board.length) {
            result.add(constructBoard(board));
            return;
        }
        
        for (int col = 0; col < board.length; col++) {
            if (isValid(board, row, col)) {
                board[row][col] = 'Q';
                backtrack(result, board, row + 1);
                board[row][col] = '.';
            }
        }
    }
    
    private boolean isValid(char[][] board, int row, int col) {
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') return false;
        }
        
        // Check diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') return false;
        }
        
        // Check anti-diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < board.length; i--, j++) {
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

**Time Complexity:** O(n!) where n is the board size
**Space Complexity:** O(n) for recursion stack

## Key Takeaways

1. Backtracking is perfect for:
   - Generating all possible combinations/permutations
   - Solving constraint satisfaction problems
   - Finding all possible solutions
   - Path finding in mazes/grids
   - Game solving

2. Common patterns:
   - Choose-explore-unchoose pattern
   - State management
   - Pruning invalid paths
   - Base case handling
   - Path restoration

3. Tips:
   - Always consider the state space
   - Use pruning to reduce search space
   - Maintain state consistency
   - Consider using visited sets for cycles
   - Think about the order of operations 