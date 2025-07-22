---
title: Trees
description: Java solutions with explanations, time and space complexity for Trees problems.
date: "June 1 2025"
---

# Tree Pattern

Trees are hierarchical data structures that consist of nodes connected by edges. They're particularly useful for:
- Representing hierarchical data
- Organizing data for quick search, insertion, and deletion
- Implementing priority queues
- Expression evaluation
- File system representation

## 1. Invert Binary Tree (Easy)

**Problem:** Given the root of a binary tree, invert the tree, and return its root.

**Solution:**
```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        
        // Swap left and right children
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        
        // Recursively invert subtrees
        invertTree(root.left);
        invertTree(root.right);
        
        return root;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree (recursion stack)

## 2. Maximum Depth of Binary Tree (Easy)

**Problem:** Given the root of a binary tree, return its maximum depth.

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

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 3. Diameter of Binary Tree (Easy)

**Problem:** Given the root of a binary tree, return the length of the diameter of the tree. The diameter is the length of the longest path between any two nodes in a tree.

**Solution:**
```java
class Solution {
    private int maxDiameter = 0;
    
    public int diameterOfBinaryTree(TreeNode root) {
        maxDepth(root);
        return maxDiameter;
    }
    
    private int maxDepth(TreeNode root) {
        if (root == null) return 0;
        
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        
        // Update diameter
        maxDiameter = Math.max(maxDiameter, leftDepth + rightDepth);
        
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 4. Balanced Binary Tree (Easy)

**Problem:** Given a binary tree, determine if it is height-balanced.

**Solution:**
```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return getHeight(root) != -1;
    }
    
    private int getHeight(TreeNode root) {
        if (root == null) return 0;
        
        int leftHeight = getHeight(root.left);
        if (leftHeight == -1) return -1;
        
        int rightHeight = getHeight(root.right);
        if (rightHeight == -1) return -1;
        
        if (Math.abs(leftHeight - rightHeight) > 1) return -1;
        
        return Math.max(leftHeight, rightHeight) + 1;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 5. Same Tree (Easy)

**Problem:** Given the roots of two binary trees p and q, write a function to check if they are the same or not.

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

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 6. Subtree of Another Tree (Easy)

**Problem:** Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot.

**Solution:**
```java
class Solution {
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) return false;
        if (isSameTree(root, subRoot)) return true;
        
        return isSubtree(root.left, subRoot) || 
               isSubtree(root.right, subRoot);
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

**Time Complexity:** O(m*n) where m and n are number of nodes in root and subRoot
**Space Complexity:** O(h) where h is the height of the tree

## 7. Lowest Common Ancestor of a Binary Search Tree (Easy)

**Problem:** Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

**Solution:**
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        }
        
        if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        
        return root;
    }
}
```

**Time Complexity:** O(h) where h is the height of the tree
**Space Complexity:** O(h) for recursion stack

## 8. Binary Tree Level Order Traversal (Medium)

**Problem:** Given the root of a binary tree, return the level order traversal of its nodes' values.

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
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            
            result.add(currentLevel);
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(n) for the queue

## 9. Binary Tree Right Side View (Medium)

**Problem:** Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

**Solution:**
```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (i == levelSize - 1) {
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

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(n) for the queue

## 10. Count Good Nodes in Binary Tree (Medium)

**Problem:** Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X. Return the number of good nodes in the binary tree.

**Solution:**
```java
class Solution {
    public int goodNodes(TreeNode root) {
        return countGoodNodes(root, Integer.MIN_VALUE);
    }
    
    private int countGoodNodes(TreeNode root, int maxSoFar) {
        if (root == null) return 0;
        
        int count = 0;
        if (root.val >= maxSoFar) {
            count = 1;
            maxSoFar = root.val;
        }
        
        count += countGoodNodes(root.left, maxSoFar);
        count += countGoodNodes(root.right, maxSoFar);
        
        return count;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 11. Validate Binary Search Tree (Medium)

**Problem:** Given the root of a binary tree, determine if it is a valid binary search tree (BST).

**Solution:**
```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return validate(root, null, null);
    }
    
    private boolean validate(TreeNode root, Integer min, Integer max) {
        if (root == null) return true;
        
        if ((min != null && root.val <= min) || 
            (max != null && root.val >= max)) {
            return false;
        }
        
        return validate(root.left, min, root.val) && 
               validate(root.right, root.val, max);
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 12. Kth Smallest Element in a BST (Medium)

**Problem:** Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

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

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 13. Construct Binary Tree from Preorder and Inorder Traversal (Medium)

**Problem:** Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

**Solution:**
```java
class Solution {
    private Map<Integer, Integer> inorderMap = new HashMap<>();
    private int preorderIndex = 0;
    
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        return buildTree(preorder, 0, inorder.length - 1);
    }
    
    private TreeNode buildTree(int[] preorder, int left, int right) {
        if (left > right) return null;
        
        int rootValue = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootValue);
        
        root.left = buildTree(preorder, left, inorderMap.get(rootValue) - 1);
        root.right = buildTree(preorder, inorderMap.get(rootValue) + 1, right);
        
        return root;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(n) for the hashmap and recursion stack

## 14. Binary Tree Maximum Path Sum (Hard)

**Problem:** A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Given the root of a binary tree, return the maximum path sum of any non-empty path.

**Solution:**
```java
class Solution {
    private int maxSum = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
    
    private int maxGain(TreeNode node) {
        if (node == null) return 0;
        
        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);
        
        int priceNewPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, priceNewPath);
        
        return node.val + Math.max(leftGain, rightGain);
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(h) where h is the height of the tree

## 15. Serialize and Deserialize Binary Tree (Hard)

**Problem:** Design an algorithm to serialize and deserialize a binary tree.

**Solution:**
```java
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "null";
        
        StringBuilder sb = new StringBuilder();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            
            if (node == null) {
                sb.append("null,");
            } else {
                sb.append(node.val).append(",");
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }
        
        return sb.toString();
    }
    
    public TreeNode deserialize(String data) {
        if (data.equals("null")) return null;
        
        String[] values = data.split(",");
        TreeNode root = new TreeNode(Integer.parseInt(values[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        for (int i = 1; i < values.length; i++) {
            TreeNode parent = queue.poll();
            
            if (!values[i].equals("null")) {
                TreeNode left = new TreeNode(Integer.parseInt(values[i]));
                parent.left = left;
                queue.offer(left);
            }
            
            if (!values[++i].equals("null")) {
                TreeNode right = new TreeNode(Integer.parseInt(values[i]));
                parent.right = right;
                queue.offer(right);
            }
        }
        
        return root;
    }
}
```

**Time Complexity:** O(n) for both serialize and deserialize
**Space Complexity:** O(n) for the queue

## Key Takeaways

1. Tree is perfect for:
   - Representing hierarchical data
   - Organizing data for quick search
   - Implementing priority queues
   - Expression evaluation
   - File system representation

2. Common patterns:
   - DFS (preorder, inorder, postorder)
   - BFS (level order)
   - Recursive solutions
   - Iterative solutions using stack/queue
   - Path finding

3. Tips:
   - Consider null cases
   - Think about traversal order
   - Use helper functions for complex operations
   - Consider space-time tradeoffs
   - Use appropriate data structures (stack/queue) 