---
title: "Meta Tagged LeetCode Problems and Solutions"
description: "Comprehensive solutions for Meta tagged LeetCode problems with Java implementations, time and space complexity analysis."
date: "2025-01-27"
order: 21
---

# Meta Tagged LeetCode Problems and Solutions

This section covers Meta tagged LeetCode problems with detailed solutions, examples, and complexity analysis. These problems are commonly asked in Meta interviews and represent important algorithmic concepts.

## 1. Minimum Remove to Make Valid Parentheses (Medium)

**Problem:** Given a string `s` of `'('` , `')'` and lowercase English characters, remove the minimum number of parentheses to make the input string valid.

**Example:**
```
Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
```

**Solution:**
```java
class Solution {
    public String minRemoveToMakeValid(String s) {
        StringBuilder sb = new StringBuilder();
        int open = 0;
        
        for (char c : s.toCharArray()) {
            if (c == '(') {
                open++;
                sb.append(c);
            } else if (c == ')') {
                if (open > 0) {
                    open--;
                    sb.append(c);
                }
            } else {
                sb.append(c);
            }
        }
        
        StringBuilder result = new StringBuilder();
        for (int i = sb.length() - 1; i >= 0; i--) {
            if (sb.charAt(i) == '(' && open > 0) {
                open--;
            } else {
                result.append(sb.charAt(i));
            }
        }
        
        return result.reverse().toString();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 2. Valid Word Abbreviation (Easy)

**Problem:** A string can be abbreviated by replacing any number of non-adjacent, non-empty substrings with their lengths.

**Example:**
```
Input: word = "internationalization", abbr = "i12iz4n"
Output: true
```

**Solution:**
```java
class Solution {
    public boolean validWordAbbreviation(String word, String abbr) {
        int i = 0, j = 0;
        
        while (i < word.length() && j < abbr.length()) {
            if (abbr.charAt(j) == '0') return false;
            
            if (Character.isDigit(abbr.charAt(j))) {
                int num = 0;
                while (j < abbr.length() && Character.isDigit(abbr.charAt(j))) {
                    num = num * 10 + (abbr.charAt(j) - '0');
                    j++;
                }
                i += num;
            } else {
                if (i >= word.length() || word.charAt(i) != abbr.charAt(j)) {
                    return false;
                }
                i++;
                j++;
            }
        }
        
        return i == word.length() && j == abbr.length();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 3. Kth Largest Element In An Array (Medium)

**Problem:** Find the kth largest element in an unsorted array.

**Example:**
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

**Solution:**
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        
        for (int num : nums) {
            pq.offer(num);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        return pq.peek();
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

---

## 4. Basic Calculator II (Medium)

**Problem:** Given a string `s` which represents an expression, evaluate this expression and return its value.

**Example:**
```
Input: s = "3+2*2"
Output: 7
```

**Solution:**
```java
class Solution {
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char sign = '+';
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            
            if (Character.isDigit(c)) {
                num = num * 10 + (c - '0');
            }
            
            if ((!Character.isDigit(c) && c != ' ') || i == s.length() - 1) {
                if (sign == '+') {
                    stack.push(num);
                } else if (sign == '-') {
                    stack.push(-num);
                } else if (sign == '*') {
                    stack.push(stack.pop() * num);
                } else if (sign == '/') {
                    stack.push(stack.pop() / num);
                }
                sign = c;
                num = 0;
            }
        }
        
        int result = 0;
        while (!stack.isEmpty()) {
            result += stack.pop();
        }
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 5. Lowest Common Ancestor of a Binary Tree III (Medium)

**Problem:** Given two nodes of a binary tree p and q, return their lowest common ancestor (LCA).

**Example:**
```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
```

**Solution:**
```java
class Solution {
    public Node lowestCommonAncestor(Node p, Node q) {
        Node a = p, b = q;
        
        while (a != b) {
            a = a == null ? q : a.parent;
            b = b == null ? p : b.parent;
        }
        
        return a;
    }
}
```

**Time Complexity:** O(h)
**Space Complexity:** O(1)

---

## 6. Binary Tree Right Side View (Medium)

**Problem:** Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

**Example:**
```
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]
```

**Solution:**
```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (i == size - 1) {
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

**Time Complexity:** O(n)
**Space Complexity:** O(w)

---

## 7. Valid Palindrome II (Easy)

**Problem:** Given a string `s`, return `true` if the `s` can be palindrome after deleting at most one character from it.

**Example:**
```
Input: s = "abca"
Output: true
```

**Solution:**
```java
class Solution {
    public boolean validPalindrome(String s) {
        int left = 0, right = s.length() - 1;
        
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return isPalindrome(s, left + 1, right) || 
                       isPalindrome(s, left, right - 1);
            }
            left++;
            right--;
        }
        
        return true;
    }
    
    private boolean isPalindrome(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 8. Random Pick with Weight (Medium)

**Problem:** You are given a 0-indexed array of positive integers `w` where `w[i]` describes the weight of the `i`th index.

**Example:**
```
Input: ["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
       [[[1,3]],[],[],[],[],[]]
Output: [null,1,1,1,1,0]
```

**Solution:**
```java
class Solution {
    private int[] prefixSum;
    private Random random;
    
    public Solution(int[] w) {
        prefixSum = new int[w.length];
        prefixSum[0] = w[0];
        for (int i = 1; i < w.length; i++) {
            prefixSum[i] = prefixSum[i-1] + w[i];
        }
        random = new Random();
    }
    
    public int pickIndex() {
        int target = random.nextInt(prefixSum[prefixSum.length - 1]) + 1;
        int left = 0, right = prefixSum.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (prefixSum[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }
}
```

**Time Complexity:** O(log n) for pickIndex
**Space Complexity:** O(n)

---

## 9. Buildings With an Ocean View (Medium)

**Problem:** There are `n` buildings in a line. You are given an integer array `heights` of size `n` that represents the heights of the buildings in the line.

**Example:**
```
Input: heights = [4,2,3,1]
Output: [0,2,3]
```

**Solution:**
```java
class Solution {
    public int[] findBuildings(int[] heights) {
        List<Integer> result = new ArrayList<>();
        int maxHeight = 0;
        
        for (int i = heights.length - 1; i >= 0; i--) {
            if (heights[i] > maxHeight) {
                result.add(i);
                maxHeight = heights[i];
            }
        }
        
        Collections.reverse(result);
        return result.stream().mapToInt(Integer::intValue).toArray();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 10. Binary Tree Vertical Order Traversal (Medium)

**Problem:** Given the `root` of a binary tree, return the vertical order traversal of its nodes' values.

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
```

**Solution:**
```java
class Solution {
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Map<Integer, List<Integer>> map = new HashMap<>();
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> colQueue = new LinkedList<>();
        
        nodeQueue.offer(root);
        colQueue.offer(0);
        
        int minCol = 0, maxCol = 0;
        
        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.poll();
            int col = colQueue.poll();
            
            map.computeIfAbsent(col, k -> new ArrayList<>()).add(node.val);
            minCol = Math.min(minCol, col);
            maxCol = Math.max(maxCol, col);
            
            if (node.left != null) {
                nodeQueue.offer(node.left);
                colQueue.offer(col - 1);
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
                colQueue.offer(col + 1);
            }
        }
        
        for (int i = minCol; i <= maxCol; i++) {
            result.add(map.get(i));
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 11. Find Peak Element (Medium)

**Problem:** A peak element is an element that is strictly greater than its neighbors.

**Example:**
```
Input: nums = [1,2,3,1]
Output: 2
```

**Solution:**
```java
class Solution {
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return left;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 12. Merge Sorted Array (Easy)

**Problem:** You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively.

**Example:**
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```

**Solution:**
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1, p = m + n - 1;
        
        while (p1 >= 0 && p2 >= 0) {
            if (nums1[p1] > nums2[p2]) {
                nums1[p] = nums1[p1];
                p1--;
            } else {
                nums1[p] = nums2[p2];
                p2--;
            }
            p--;
        }
        
        while (p2 >= 0) {
            nums1[p] = nums2[p2];
            p2--;
            p--;
        }
    }
}
```

**Time Complexity:** O(m + n)
**Space Complexity:** O(1)

---

## 13. Diameter of Binary Tree (Easy)

**Problem:** Given the `root` of a binary tree, return the length of the diameter of the tree.

**Example:**
```
Input: root = [1,2,3,4,5]
Output: 3
```

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
        
        maxDiameter = Math.max(maxDiameter, leftDepth + rightDepth);
        
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 14. Merge Intervals (Medium)

**Problem:** Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals.

**Example:**
```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

**Solution:**
```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> result = new ArrayList<>();
        int[] current = intervals[0];
        
        for (int[] interval : intervals) {
            if (current[1] >= interval[0]) {
                current[1] = Math.max(current[1], interval[1]);
            } else {
                result.add(current);
                current = interval;
            }
        }
        
        result.add(current);
        return result.toArray(new int[result.size()][]);
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

## 15. Valid Palindrome (Easy)

**Problem:** Given a string `s`, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

**Example:**
```
Input: s = "A man, a plan, a canal: Panama"
Output: true
```

**Solution:**
```java
class Solution {
    public boolean isPalindrome(String s) {
        int left = 0, right = s.length() - 1;
        
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            
            if (Character.toLowerCase(s.charAt(left)) != 
                Character.toLowerCase(s.charAt(right))) {
                return false;
            }
            left++;
            right--;
        }
        
        return true;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 16. Sum Root to Leaf Numbers (Medium)

**Problem:** You are given the `root` of a binary tree containing digits from `0` to `9` only.

**Example:**
```
Input: root = [1,2,3]
Output: 25
```

**Solution:**
```java
class Solution {
    private int totalSum = 0;
    
    public int sumNumbers(TreeNode root) {
        dfs(root, 0);
        return totalSum;
    }
    
    private void dfs(TreeNode root, int currentSum) {
        if (root == null) return;
        
        currentSum = currentSum * 10 + root.val;
        
        if (root.left == null && root.right == null) {
            totalSum += currentSum;
            return;
        }
        
        dfs(root.left, currentSum);
        dfs(root.right, currentSum);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 17. Top K Frequent Elements (Medium)

**Problem:** Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

**Example:**
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

**Solution:**
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> pq = 
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        
        for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.poll().getKey();
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(n)

---

## 18. Subarray Sum Equals K (Medium)

**Problem:** Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals to `k`.

**Example:**
```
Input: nums = [1,1,1], k = 2
Output: 2
```

**Solution:**
```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        
        int sum = 0, count = 0;
        
        for (int num : nums) {
            sum += num;
            if (map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        
        return count;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 19. Best Time to Buy and Sell Stock (Easy)

**Problem:** You are given an array `prices` where `prices[i]` is the price of a given stock on the `i`th day.

**Example:**
```
Input: prices = [7,1,5,3,6,4]
Output: 5
```

**Solution:**
```java
class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        
        for (int price : prices) {
            minPrice = Math.min(minPrice, price);
            maxProfit = Math.max(maxProfit, price - minPrice);
        }
        
        return maxProfit;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 20. Range Sum of BST (Easy)

**Problem:** Given the `root` node of a binary search tree and two integers `low` and `high`, return the sum of values of all nodes with a value in the inclusive range `[low, high]`.

**Example:**
```
Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
Output: 32
```

**Solution:**
```java
class Solution {
    private int sum = 0;
    
    public int rangeSumBST(TreeNode root, int low, int high) {
        dfs(root, low, high);
        return sum;
    }
    
    private void dfs(TreeNode root, int low, int high) {
        if (root == null) return;
        
        if (root.val >= low && root.val <= high) {
            sum += root.val;
        }
        
        if (root.val > low) {
            dfs(root.left, low, high);
        }
        
        if (root.val < high) {
            dfs(root.right, low, high);
        }
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

---

## 21. K Closest Points to Origin (Medium)

**Problem:** Given an array of `points` where `points[i] = [xi, yi]` represents a point on the X-Y plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

**Example:**
```
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
```

**Solution:**
```java
class Solution {
    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> 
            (b[0] * b[0] + b[1] * b[1]) - (a[0] * a[0] + a[1] * a[1]));
        
        for (int[] point : points) {
            pq.offer(point);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        int[][] result = new int[k][2];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.poll();
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

---

## 22. Valid Parentheses (Easy)

**Problem:** Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

**Example:**
```
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false
```

**Solution:**
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) return false;
                
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 23. Merge K Sorted Lists (Hard)

**Problem:** You are given an array of `k` linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

**Example:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

**Solution:**
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            current.next = node;
            current = current.next;
            
            if (node.next != null) {
                pq.offer(node.next);
            }
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

---

## 24. Interval List Intersections (Medium)

**Problem:** You are given two lists of closed intervals, `firstList` and `secondList`, where `firstList[i] = [starti, endi]` and `secondList[j] = [startj, endj]`. Each list of intervals is pairwise disjoint and in sorted order.

**Example:**
```
Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**Solution:**
```java
class Solution {
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> result = new ArrayList<>();
        int i = 0, j = 0;
        
        while (i < firstList.length && j < secondList.length) {
            int start = Math.max(firstList[i][0], secondList[j][0]);
            int end = Math.min(firstList[i][1], secondList[j][1]);
            
            if (start <= end) {
                result.add(new int[]{start, end});
            }
            
            if (firstList[i][1] < secondList[j][1]) {
                i++;
            } else {
                j++;
            }
        }
        
        return result.toArray(new int[result.size()][]);
    }
}
```

**Time Complexity:** O(m + n)
**Space Complexity:** O(min(m, n))

---

## 25. Clone Graph (Medium)

**Problem:** Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

**Example:**
```
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
```

**Solution:**
```java
class Solution {
    private Map<Node, Node> visited = new HashMap<>();
    
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        
        if (visited.containsKey(node)) {
            return visited.get(node);
        }
        
        Node cloneNode = new Node(node.val);
        visited.put(node, cloneNode);
        
        for (Node neighbor : node.neighbors) {
            cloneNode.neighbors.add(cloneGraph(neighbor));
        }
        
        return cloneNode;
    }
}
```

**Time Complexity:** O(V + E)
**Space Complexity:** O(V)

---

## 26. Shortest Path in Binary Matrix (Medium)

**Problem:** Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from the top-left corner `(0, 0)` to the bottom-right corner `(n - 1, n - 1)`.

**Example:**
```
Input: grid = [[0,1],[1,0]]
Output: 2
```

**Solution:**
```java
class Solution {
    public int shortestPathBinaryMatrix(int[][] grid) {
        if (grid[0][0] == 1) return -1;
        
        int n = grid.length;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0, 0, 1});
        grid[0][0] = 1;
        
        int[][] directions = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
        
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int row = current[0], col = current[1], distance = current[2];
            
            if (row == n - 1 && col == n - 1) {
                return distance;
            }
            
            for (int[] dir : directions) {
                int newRow = row + dir[0];
                int newCol = col + dir[1];
                
                if (newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && grid[newRow][newCol] == 0) {
                    grid[newRow][newCol] = 1;
                    queue.offer(new int[]{newRow, newCol, distance + 1});
                }
            }
        }
        
        return -1;
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(n²)

---

## 27. Pow(x, n) (Medium)

**Problem:** Implement `pow(x, n)`, which calculates `x` raised to the power `n` (i.e., `xⁿ`).

**Example:**
```
Input: x = 2.00000, n = 10
Output: 1024.00000
```

**Solution:**
```java
class Solution {
    public double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;
        
        double half = myPow(x, n / 2);
        if (n % 2 == 0) {
            return half * half;
        } else {
            return half * half * (n > 0 ? x : 1 / x);
        }
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(log n)

---

## 28. Minimum Window Substring (Hard)

**Problem:** Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window.

**Example:**
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

**Solution:**
```java
class Solution {
    public String minWindow(String s, String t) {
        if (s.length() < t.length()) return "";
        
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> have = new HashMap<>();
        
        for (char c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }
        
        int left = 0, right = 0;
        int minLen = Integer.MAX_VALUE;
        int minStart = 0;
        int required = need.size();
        int formed = 0;
        
        while (right < s.length()) {
            char c = s.charAt(right);
            have.put(c, have.getOrDefault(c, 0) + 1);
            
            if (need.containsKey(c) && have.get(c).equals(need.get(c))) {
                formed++;
            }
            
            while (left <= right && formed == required) {
                char leftChar = s.charAt(left);
                
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }
                
                have.put(leftChar, have.get(leftChar) - 1);
                if (need.containsKey(leftChar) && have.get(leftChar) < need.get(leftChar)) {
                    formed--;
                }
                left++;
            }
            right++;
        }
        
        return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(k)

---

## 29. Copy List With Random Pointer (Medium)

**Problem:** A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

**Example:**
```
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**Solution:**
```java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        
        // Step 1: Create copy nodes and insert them after original nodes
        Node current = head;
        while (current != null) {
            Node copy = new Node(current.val);
            copy.next = current.next;
            current.next = copy;
            current = copy.next;
        }
        
        // Step 2: Set random pointers for copy nodes
        current = head;
        while (current != null) {
            if (current.random != null) {
                current.next.random = current.random.next;
            }
            current = current.next.next;
        }
        
        // Step 3: Separate original and copy lists
        Node dummy = new Node(0);
        Node copyCurrent = dummy;
        current = head;
        
        while (current != null) {
            copyCurrent.next = current.next;
            copyCurrent = copyCurrent.next;
            current.next = current.next.next;
            current = current.next;
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 30. Longest Common Prefix (Easy)

**Problem:** Write a function to find the longest common prefix string amongst an array of strings.

**Example:**
```
Input: strs = ["flower","flow","flight"]
Output: "fl"
```

**Solution:**
```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.isEmpty()) return "";
            }
        }
        
        return prefix;
    }
}
```

**Time Complexity:** O(S)
**Space Complexity:** O(1)

---

## 31. Find First and Last Position of Element in Sorted Array (Medium)

**Problem:** Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

**Example:**
```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Solution:**
```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1, -1};
        
        result[0] = findFirst(nums, target);
        result[1] = findLast(nums, target);
        
        return result;
    }
    
    private int findFirst(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                result = mid;
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    private int findLast(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                result = mid;
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1)

---

## 32. Accounts Merge (Medium)

**Problem:** Given a list of `accounts` where each element `accounts[i]` is a list of strings, where the first element `accounts[i][0]` is a name, and the rest of the elements are emails representing emails of the account.

**Example:**
```
Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["John","johnnybravo@mail.com"],["Mary","mary@mail.com"]]
```

**Solution:**
```java
class Solution {
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, String> emailToName = new HashMap<>();
        Map<String, Set<String>> graph = new HashMap<>();
        
        // Build graph
        for (List<String> account : accounts) {
            String name = account.get(0);
            for (int i = 1; i < account.size(); i++) {
                String email = account.get(i);
                emailToName.put(email, name);
                graph.computeIfAbsent(email, k -> new HashSet<>());
                
                if (i == 1) continue;
                String firstEmail = account.get(1);
                graph.get(email).add(firstEmail);
                graph.get(firstEmail).add(email);
            }
        }
        
        // DFS to find connected components
        Set<String> visited = new HashSet<>();
        List<List<String>> result = new ArrayList<>();
        
        for (String email : emailToName.keySet()) {
            if (!visited.contains(email)) {
                List<String> component = new ArrayList<>();
                dfs(email, graph, visited, component);
                Collections.sort(component);
                component.add(0, emailToName.get(email));
                result.add(component);
            }
        }
        
        return result;
    }
    
    private void dfs(String email, Map<String, Set<String>> graph, Set<String> visited, List<String> component) {
        visited.add(email);
        component.add(email);
        
        for (String neighbor : graph.get(email)) {
            if (!visited.contains(neighbor)) {
                dfs(neighbor, graph, visited, component);
            }
        }
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

---

## 33. Squares of a Sorted Array (Easy)

**Problem:** Given an integer array `nums` sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

**Example:**
```
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
```

**Solution:**
```java
class Solution {
    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        int left = 0, right = n - 1;
        int index = n - 1;
        
        while (left <= right) {
            int leftSquare = nums[left] * nums[left];
            int rightSquare = nums[right] * nums[right];
            
            if (leftSquare > rightSquare) {
                result[index] = leftSquare;
                left++;
            } else {
                result[index] = rightSquare;
                right--;
            }
            index--;
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## 34. LRU Cache (Medium)

**Problem:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

**Example:**
```
Input: ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
       [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output: [null, null, null, 1, null, -1, null, -1, 3, 4]
```

**Solution:**
```java
class LRUCache {
    private Map<Integer, Node> cache;
    private Node head, tail;
    private int capacity;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        cache = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            remove(node);
            add(node);
            return node.value;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            remove(cache.get(key));
        }
        
        Node newNode = new Node(key, value);
        cache.put(key, newNode);
        add(newNode);
        
        if (cache.size() > capacity) {
            Node lru = head.next;
            remove(lru);
            cache.remove(lru.key);
        }
    }
    
    private void add(Node node) {
        node.prev = tail.prev;
        node.next = tail;
        tail.prev.next = node;
        tail.prev = node;
    }
    
    private void remove(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    class Node {
        int key, value;
        Node prev, next;
        
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
}
```

**Time Complexity:** O(1)
**Space Complexity:** O(capacity)

---

## 35. Remove Nth Node From End of List (Medium)

**Problem:** Given the `head` of a linked list, remove the `n`th node from the end of the list and return its head.

**Example:**
```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```

**Solution:**
```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode first = dummy;
        ListNode second = dummy;
        
        for (int i = 0; i <= n; i++) {
            first = first.next;
        }
        
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        
        second.next = second.next.next;
        return dummy.next;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 36. Palindromic Substrings (Medium)

**Problem:** Given a string `s`, return the number of palindromic substrings in it.

**Example:**
```
Input: s = "abc"
Output: 3
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

## 37. Max Consecutive Ones III (Medium)

**Problem:** Given a binary array `nums` and an integer `k`, return the maximum number of consecutive `1`'s in the array if you can flip at most `k` `0`'s.

**Example:**
```
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
```

**Solution:**
```java
class Solution {
    public int longestOnes(int[] nums, int k) {
        int left = 0, right = 0;
        int zeros = 0;
        int maxLength = 0;
        
        while (right < nums.length) {
            if (nums[right] == 0) {
                zeros++;
            }
            
            while (zeros > k) {
                if (nums[left] == 0) {
                    zeros--;
                }
                left++;
            }
            
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }
        
        return maxLength;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## 38. Custom Sort String (Medium)

**Problem:** You are given two strings `order` and `s`. All the characters of `order` are unique and were sorted in some custom order previously.

**Example:**
```
Input: order = "cba", s = "abcd"
Output: "cbad"
```

**Solution:**
```java
class Solution {
    public String customSortString(String order, String s) {
        int[] count = new int[26];
        
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        
        StringBuilder result = new StringBuilder();
        
        for (char c : order.toCharArray()) {
            while (count[c - 'a'] > 0) {
                result.append(c);
                count[c - 'a']--;
            }
        }
        
        for (int i = 0; i < 26; i++) {
            while (count[i] > 0) {
                result.append((char) (i + 'a'));
                count[i]--;
            }
        }
        
        return result.toString();
    }
}
```

---

## 38. Evaluate Reverse Polish Notation (Medium)

**Problem:** You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation. Evaluate the expression. Return an integer that represents the value of the expression.

**Example:**
```
Input: tokens = ["2","1","+","3","*"]
Output: 9
```

**Solution:**
```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (token.equals("-")) {
                int a = stack.pop();
                int b = stack.pop();
                stack.push(b - a);
            } else if (token.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (token.equals("/")) {
                int a = stack.pop();
                int b = stack.pop();
                stack.push(b / a);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## Summary

These Meta tagged problems cover essential algorithmic concepts including:

- **Arrays & Hashing:** Kth Largest Element, Top K Frequent Elements, Subarray Sum Equals K
- **Two Pointers:** Valid Palindrome, Valid Palindrome II
- **Sliding Window:** Max Consecutive Ones III, Minimum Window Substring
- **Stack:** Valid Parentheses, Minimum Remove to Make Valid Parentheses
- **Linked Lists:** Merge K Sorted Lists, Copy List With Random Pointer, Remove Nth Node From End of List
- **Dynamic Programming:** Palindromic Substrings
- **Backtracking:** Sum Root to Leaf Numbers
- **Graph:** Clone Graph, Shortest Path in Binary Matrix, Accounts Merge
- **Design:** LRU Cache
- **String:** Valid Word Abbreviation, Longest Common Prefix, Custom Sort String
- **Greedy:** K Closest Points to Origin
- **Binary Search:** Find First And Last Position of Element In Sorted Array, Pow(x, n), Find Peak Element
- **Tree:** Binary Tree Right Side View, Binary Tree Vertical Order Traversal, Diameter of Binary Tree, Range Sum of BST, Lowest Common Ancestor
- **Math:** Basic Calculator II, Random Pick with Weight
- **Matrix:** Buildings With an Ocean View, Squares of a Sorted Array
- **Sorting:** Merge Sorted Array, Merge Intervals
- **Interval:** Interval List Intersections

Each solution includes detailed explanations, time and space complexity analysis, and follows best practices for coding interviews. These problems are commonly asked in Meta technical interviews and represent the core algorithmic concepts that candidates should master. 