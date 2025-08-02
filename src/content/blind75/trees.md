---
title: Trees
description: Java solutions with explanations, time and space complexity for Trees problems from Blind 75.
date: "June 1 2025"
order: 7
---

# Trees

This section covers problems involving binary trees, binary search trees, and tree traversal algorithms.

## 1. Invert Binary Tree (Easy)

**Problem:** Given the `root` of a binary tree, invert the tree, and return its root.

**Example:**
```
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
```

**Solution:**
```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        
        // Invert left and right subtrees
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        
        // Swap left and right children
        root.left = right;
        root.right = left;
        
        return root;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h) where h is the height of the tree

---

## 2. Maximum Depth of Binary Tree (Easy)

**Problem:** Given the `root` of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: 3
```

**Solution:**
```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 3. Same Tree (Easy)

**Problem:** Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

**Example:**
```
Input: p = [1,2,3], q = [1,2,3]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        
        return p.val == q.val && 
               isSameTree(p.left, q.left) && 
               isSameTree(p.right, q.right);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 4. Subtree of Another Tree (Easy)

**Problem:** Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree of `root` with the same structure and node values of `subRoot` and `false` otherwise.

**Example:**
```
Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (subRoot == null) return true;
        if (root == null) return false;
        
        if (isSameTree(root, subRoot)) return true;
        
        return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }
    
    private boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        
        return p.val == q.val && 
               isSameTree(p.left, q.left) && 
               isSameTree(p.right, q.right);
    }
}
```

**Time Complexity:** O(n * m) where n and m are the number of nodes in root and subRoot
**Space Complexity:** O(h)

---

## 5. Lowest Common Ancestor of a Binary Search Tree (Medium)

**Problem:** Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

**Example:**
```
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
```

**Solution:**
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        
        // If both p and q are less than root, LCA is in left subtree
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        }
        
        // If both p and q are greater than root, LCA is in right subtree
        if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        
        // If p and q are on different sides, root is LCA
        return root;
    }
}
```

**Time Complexity:** O(h)
**Space Complexity:** O(h)

---

## 6. Binary Tree Level Order Traversal (Medium)

**Problem:** Given the `root` of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

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

## 7. Validate Binary Search Tree (Medium)

**Problem:** Given the `root` of a binary tree, determine if it is a valid binary search tree (BST).

**Example:**
```
Input: root = [2,1,3]
Output: true
```

**Solution:**
```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, null, null);
    }
    
    private boolean isValidBST(TreeNode root, Integer min, Integer max) {
        if (root == null) return true;
        
        if ((min != null && root.val <= min) || 
            (max != null && root.val >= max)) {
            return false;
        }
        
        return isValidBST(root.left, min, root.val) && 
               isValidBST(root.right, root.val, max);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 8. Kth Smallest Element in a BST (Medium)

**Problem:** Given the `root` of a binary search tree, and an integer `k`, return the `kth` smallest value (1-indexed) of all the values of the nodes in the tree.

**Example:**
```
Input: root = [3,1,4,null,2], k = 1
Output: 1
```

**Solution:**
```java
class Solution {
    private int count = 0;
    private int result = 0;
    
    public int kthSmallest(TreeNode root, int k) {
        inorder(root, k);
        return result;
    }
    
    private void inorder(TreeNode root, int k) {
        if (root == null) return;
        
        inorder(root.left, k);
        
        count++;
        if (count == k) {
            result = root.val;
            return;
        }
        
        inorder(root.right, k);
    }
}
```

**Time Complexity:** O(k)
**Space Complexity:** O(h)

---

## 9. Construct Binary Tree from Preorder and Inorder Traversal (Medium)

**Problem:** Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.

**Example:**
```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

**Solution:**
```java
class Solution {
    private int preorderIndex = 0;
    private Map<Integer, Integer> inorderMap = new HashMap<>();
    
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        return buildTreeHelper(preorder, 0, inorder.length - 1);
    }
    
    private TreeNode buildTreeHelper(int[] preorder, int left, int right) {
        if (left > right) return null;
        
        int rootValue = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootValue);
        
        int inorderIndex = inorderMap.get(rootValue);
        
        root.left = buildTreeHelper(preorder, left, inorderIndex - 1);
        root.right = buildTreeHelper(preorder, inorderIndex + 1, right);
        
        return root;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 10. Binary Tree Maximum Path Sum (Hard)

**Problem:** A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, return the maximum path sum of any non-empty path.

**Example:**
```
Input: root = [1,2,3]
Output: 6
```

**Solution:**
```java
class Solution {
    private int maxSum = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
    
    private int maxGain(TreeNode root) {
        if (root == null) return 0;
        
        int leftGain = Math.max(maxGain(root.left), 0);
        int rightGain = Math.max(maxGain(root.right), 0);
        
        int priceNewPath = root.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, priceNewPath);
        
        return root.val + Math.max(leftGain, rightGain);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 11. Serialize and Deserialize Binary Tree (Hard)

**Problem:** Design an algorithm to serialize and deserialize a binary tree.

**Example:**
```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
```

**Solution:**
```java
public class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "null";
        
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }
    
    // Decodes your encoded data to tree.
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

## Key Takeaways

1. **Tree Traversal**: Master inorder, preorder, and postorder traversals
2. **Recursion**: Most tree problems can be solved recursively
3. **BST Properties**: Use BST properties for efficient searching
4. **Level Order**: Use BFS for level-order traversal
5. **Path Problems**: Consider both single path and combined path scenarios
6. **Serialization**: Use special markers for null nodes 