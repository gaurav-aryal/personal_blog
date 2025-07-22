---
title: Tries
description: Java solutions with explanations, time and space complexity for Tries problems.
date: "June 1 2025"
---

# Trie Pattern

Tries (Prefix Trees) are tree-like data structures used to store and retrieve strings. They're particularly useful for:
- Autocomplete features
- Spell checkers
- IP routing tables
- Dictionary implementations
- Prefix-based searches

## 1. Implement Trie (Prefix Tree) (Medium)

**Problem:** Implement a Trie data structure with insert, search, and startsWith methods.

**Solution:**
```java
class Trie {
    private class TrieNode {
        private TrieNode[] children;
        private boolean isEndOfWord;
        
        public TrieNode() {
            children = new TrieNode[26];
            isEndOfWord = false;
        }
    }
    
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode current = root;
        
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (current.children[index] == null) {
                current.children[index] = new TrieNode();
            }
            current = current.children[index];
        }
        
        current.isEndOfWord = true;
    }
    
    public boolean search(String word) {
        TrieNode node = searchPrefix(word);
        return node != null && node.isEndOfWord;
    }
    
    public boolean startsWith(String prefix) {
        return searchPrefix(prefix) != null;
    }
    
    private TrieNode searchPrefix(String prefix) {
        TrieNode current = root;
        
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (current.children[index] == null) {
                return null;
            }
            current = current.children[index];
        }
        
        return current;
    }
}
```

**Time Complexity:**
- insert: O(m) where m is the word length
- search: O(m) where m is the word length
- startsWith: O(m) where m is the prefix length
**Space Complexity:** O(ALPHABET_SIZE * N * M) where N is the number of words and M is the average word length

## 2. Design Add and Search Words Data Structure (Medium)

**Problem:** Design a data structure that supports adding new words and finding if a string matches any previously added string. The '.' character can match any single character.

**Solution:**
```java
class WordDictionary {
    private class TrieNode {
        private TrieNode[] children;
        private boolean isEndOfWord;
        
        public TrieNode() {
            children = new TrieNode[26];
            isEndOfWord = false;
        }
    }
    
    private TrieNode root;
    
    public WordDictionary() {
        root = new TrieNode();
    }
    
    public void addWord(String word) {
        TrieNode current = root;
        
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (current.children[index] == null) {
                current.children[index] = new TrieNode();
            }
            current = current.children[index];
        }
        
        current.isEndOfWord = true;
    }
    
    public boolean search(String word) {
        return searchInNode(word, 0, root);
    }
    
    private boolean searchInNode(String word, int index, TrieNode node) {
        if (node == null) return false;
        if (index == word.length()) return node.isEndOfWord;
        
        char c = word.charAt(index);
        if (c == '.') {
            for (TrieNode child : node.children) {
                if (child != null && searchInNode(word, index + 1, child)) {
                    return true;
                }
            }
            return false;
        } else {
            return searchInNode(word, index + 1, node.children[c - 'a']);
        }
    }
}
```

**Time Complexity:**
- addWord: O(m) where m is the word length
- search: O(26^m) in worst case where m is the word length (when word contains only '.' characters)
**Space Complexity:** O(ALPHABET_SIZE * N * M) where N is the number of words and M is the average word length

## 3. Word Search II (Hard)

**Problem:** Given an m x n board of characters and a list of strings words, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring.

**Solution:**
```java
class Solution {
    private class TrieNode {
        private TrieNode[] children;
        private String word;
        
        public TrieNode() {
            children = new TrieNode[26];
            word = null;
        }
    }
    
    private TrieNode root;
    private List<String> result;
    
    public List<String> findWords(char[][] board, String[] words) {
        root = new TrieNode();
        result = new ArrayList<>();
        
        // Build Trie
        for (String word : words) {
            TrieNode current = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (current.children[index] == null) {
                    current.children[index] = new TrieNode();
                }
                current = current.children[index];
            }
            current.word = word;
        }
        
        // Search in board
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, i, j, root);
            }
        }
        
        return result;
    }
    
    private void dfs(char[][] board, int i, int j, TrieNode node) {
        char c = board[i][j];
        if (c == '#' || node.children[c - 'a'] == null) return;
        
        node = node.children[c - 'a'];
        if (node.word != null) {
            result.add(node.word);
            node.word = null; // Avoid duplicates
        }
        
        board[i][j] = '#'; // Mark as visited
        
        if (i > 0) dfs(board, i - 1, j, node);
        if (j > 0) dfs(board, i, j - 1, node);
        if (i < board.length - 1) dfs(board, i + 1, j, node);
        if (j < board[0].length - 1) dfs(board, i, j + 1, node);
        
        board[i][j] = c; // Backtrack
    }
}
```

**Time Complexity:** O(m * n * 4^L) where m and n are board dimensions and L is the maximum word length
**Space Complexity:** O(ALPHABET_SIZE * N * M) where N is the number of words and M is the average word length

## Key Takeaways

1. Trie is perfect for:
   - String operations
   - Prefix-based searches
   - Autocomplete features
   - Spell checking
   - Dictionary implementations

2. Common patterns:
   - Node structure with children array/map
   - End of word marker
   - Prefix-based traversal
   - Backtracking with DFS
   - Character to index mapping

3. Tips:
   - Consider space optimization (using HashMap instead of array)
   - Handle special characters (like '.' in wildcard matching)
   - Use backtracking for board-based problems
   - Consider memory usage for large datasets
   - Implement proper cleanup for memory management 