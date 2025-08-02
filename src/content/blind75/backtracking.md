---
title: Backtracking
description: Java solutions with explanations, time and space complexity for Backtracking problems from Blind 75.
date: "June 1 2025"
order: 9
---

# Backtracking

This section covers problems that can be solved using backtracking algorithms.

## 1. Combination Sum (Medium)

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
        backtrack(candidates, target, 0, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrack(int[] candidates, int target, int start, 
                          List<Integer> current, List<List<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        if (target < 0) return;
        
        for (int i = start; i < candidates.length; i++) {
            current.add(candidates[i]);
            backtrack(candidates, target - candidates[i], i, current, result);
            current.remove(current.size() - 1);
        }
    }
}
```

**Time Complexity:** O(n^(target/min))
**Space Complexity:** O(target/min)

---

## 2. Word Search (Medium)

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
        int m = board.length, n = board[0].length;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (backtrack(board, word, 0, i, j)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private boolean backtrack(char[][] board, String word, int index, int i, int j) {
        if (index == word.length()) return true;
        
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
            board[i][j] != word.charAt(index)) {
            return false;
        }
        
        char temp = board[i][j];
        board[i][j] = '#';
        
        boolean result = backtrack(board, word, index + 1, i + 1, j) ||
                        backtrack(board, word, index + 1, i - 1, j) ||
                        backtrack(board, word, index + 1, i, j + 1) ||
                        backtrack(board, word, index + 1, i, j - 1);
        
        board[i][j] = temp;
        return result;
    }
}
```

**Time Complexity:** O(m * n * 4^L) where L is the length of word
**Space Complexity:** O(L)

## Key Takeaways

1. **State Management**: Track current state and backtrack when needed
2. **Pruning**: Use early termination to improve performance
3. **Visited Tracking**: Mark visited cells to avoid cycles
4. **Recursion**: Use recursion for natural backtracking implementation 