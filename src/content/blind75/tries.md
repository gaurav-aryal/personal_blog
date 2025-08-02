---
title: Tries
description: Java solutions with explanations, time and space complexity for Tries problems from Blind 75.
date: "June 1 2025"
order: 10
---

# Tries

This section covers problems involving trie data structures for efficient string operations.

## 1. Implement Trie (Prefix Tree) (Medium)

**Problem:** A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings.

Implement the Trie class:
- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string word into the trie.
- `boolean search(String word)` Returns true if the string word is in the trie, and false otherwise.
- `boolean startsWith(String prefix)` Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

**Example:**
```
Input: ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output: [null, null, true, false, true, null, true]
```

**Solution:**
```java
class Trie {
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            if (!current.children.containsKey(c)) {
                current.children.put(c, new TrieNode());
            }
            current = current.children.get(c);
        }
        current.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            if (!current.children.containsKey(c)) {
                return false;
            }
            current = current.children.get(c);
        }
        return current.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode current = root;
        for (char c : prefix.toCharArray()) {
            if (!current.children.containsKey(c)) {
                return false;
            }
            current = current.children.get(c);
        }
        return true;
    }
}

class TrieNode {
    Map<Character, TrieNode> children;
    boolean isEnd;
    
    public TrieNode() {
        children = new HashMap<>();
        isEnd = false;
    }
}
```

**Time Complexity:** O(m) for insert, search, and startsWith where m is the length of the string
**Space Complexity:** O(m) for insert, O(1) for search and startsWith

---

## 2. Design Add and Search Words Data Structure (Medium)

**Problem:** Design a data structure that supports adding new words and finding if a string matches any previously added string.

**Example:**
```
Input: ["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output: [null,null,null,null,false,true,true,true]
```

**Solution:**
```java
class WordDictionary {
    private TrieNode root;
    
    public WordDictionary() {
        root = new TrieNode();
    }
    
    public void addWord(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            if (!current.children.containsKey(c)) {
                current.children.put(c, new TrieNode());
            }
            current = current.children.get(c);
        }
        current.isEnd = true;
    }
    
    public boolean search(String word) {
        return searchHelper(word, 0, root);
    }
    
    private boolean searchHelper(String word, int index, TrieNode node) {
        if (index == word.length()) {
            return node.isEnd;
        }
        
        char c = word.charAt(index);
        if (c == '.') {
            for (TrieNode child : node.children.values()) {
                if (searchHelper(word, index + 1, child)) {
                    return true;
                }
            }
            return false;
        } else {
            if (!node.children.containsKey(c)) {
                return false;
            }
            return searchHelper(word, index + 1, node.children.get(c));
        }
    }
}
```

**Time Complexity:** O(m) for addWord, O(m * 26^d) for search where d is the number of dots
**Space Complexity:** O(m) for addWord, O(m) for search

---

## 3. Word Search II (Hard)

**Problem:** Given an `m x n` `board` of characters and a list of strings `words`, return all words on the board.

**Example:**
```
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
```

**Solution:**
```java
class Solution {
    private TrieNode root;
    private Set<String> result;
    private int[][] directions = {{0,1}, {1,0}, {0,-1}, {-1,0}};
    
    public List<String> findWords(char[][] board, String[] words) {
        root = new TrieNode();
        result = new HashSet<>();
        
        // Build trie
        for (String word : words) {
            insert(word);
        }
        
        int m = board.length, n = board[0].length;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(board, i, j, root, "");
            }
        }
        
        return new ArrayList<>(result);
    }
    
    private void dfs(char[][] board, int i, int j, TrieNode node, String word) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
            board[i][j] == '#' || !node.children.containsKey(board[i][j])) {
            return;
        }
        
        char c = board[i][j];
        node = node.children.get(c);
        word += c;
        
        if (node.isEnd) {
            result.add(word);
        }
        
        board[i][j] = '#';
        
        for (int[] dir : directions) {
            dfs(board, i + dir[0], j + dir[1], node, word);
        }
        
        board[i][j] = c;
    }
    
    private void insert(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            if (!current.children.containsKey(c)) {
                current.children.put(c, new TrieNode());
            }
            current = current.children.get(c);
        }
        current.isEnd = true;
    }
}
```

**Time Complexity:** O(m * n * 4^L) where L is the maximum word length
**Space Complexity:** O(k * L) where k is the number of words

## Key Takeaways

1. **Prefix Matching**: Tries excel at prefix-based operations
2. **Wildcard Support**: Handle wildcards with recursive backtracking
3. **Memory Efficiency**: Use hash maps for character mapping
4. **Word Search**: Combine trie with DFS for word search problems 