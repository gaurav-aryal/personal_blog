---
title: Miscellaneous Data Structures & Algorithms
description: Java solutions with explanations, time and space complexity for Miscellaneous Data Structures & Algorithms problems.
date: "June 1 2025"
---

# Miscellaneous DSA Patterns

This section covers important data structures and algorithms that are fundamental to computer science but not typically covered in other patterns:
- Union Find (Disjoint Set)
- Graph Algorithms (Dijkstra's, Bellman-Ford)
- String Matching (KMP)
- Advanced Tree Structures (Segment Tree, Fenwick Tree)

## 1. Union Find (Disjoint Set)

**Problem:** Implement a Union-Find data structure with path compression and union by rank.

**Solution:**
```java
class UnionFind {
    private int[] parent;
    private int[] rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    public void union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return;
        
        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
}
```

**Time Complexity:**
- Find: O(α(n)) amortized
- Union: O(α(n)) amortized
**Space Complexity:** O(n)

## 2. Dijkstra's Algorithm

**Problem:** Implement Dijkstra's algorithm to find the shortest path in a weighted graph.

**Solution:**
```java
class Solution {
    public int[] dijkstra(int[][] graph, int source) {
        int n = graph.length;
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[source] = 0;
        
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        pq.offer(new int[]{source, 0});
        
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int node = current[0];
            int distance = current[1];
            
            if (distance > dist[node]) continue;
            
            for (int[] neighbor : graph[node]) {
                int nextNode = neighbor[0];
                int weight = neighbor[1];
                
                if (dist[node] + weight < dist[nextNode]) {
                    dist[nextNode] = dist[node] + weight;
                    pq.offer(new int[]{nextNode, dist[nextNode]});
                }
            }
        }
        
        return dist;
    }
}
```

**Time Complexity:** O((V + E) log V)
**Space Complexity:** O(V)

## 3. BFS on Graph

**Problem:** Implement Breadth-First Search on a graph.

**Solution:**
```java
class Solution {
    public void bfs(List<List<Integer>> graph, int start) {
        int n = graph.size();
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        
        visited[start] = true;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int node = queue.poll();
            System.out.print(node + " ");
            
            for (int neighbor : graph.get(node)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
    }
}
```

**Time Complexity:** O(V + E)
**Space Complexity:** O(V)

## 4. Bellman-Ford Algorithm

**Problem:** Implement Bellman-Ford algorithm to find shortest paths with negative edge weights.

**Solution:**
```java
class Solution {
    public int[] bellmanFord(int[][] edges, int n, int source) {
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[source] = 0;
        
        // Relax all edges n-1 times
        for (int i = 0; i < n - 1; i++) {
            for (int[] edge : edges) {
                int u = edge[0], v = edge[1], w = edge[2];
                if (dist[u] != Integer.MAX_VALUE && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                }
            }
        }
        
        // Check for negative cycles
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            if (dist[u] != Integer.MAX_VALUE && dist[u] + w < dist[v]) {
                throw new RuntimeException("Negative cycle detected");
            }
        }
        
        return dist;
    }
}
```

**Time Complexity:** O(VE)
**Space Complexity:** O(V)

## 5. KMP Algorithm

**Problem:** Implement the Knuth-Morris-Pratt algorithm for string matching.

**Solution:**
```java
class Solution {
    public int kmp(String text, String pattern) {
        int[] lps = computeLPS(pattern);
        int i = 0; // index for text
        int j = 0; // index for pattern
        
        while (i < text.length()) {
            if (pattern.charAt(j) == text.charAt(i)) {
                i++;
                j++;
            }
            
            if (j == pattern.length()) {
                return i - j; // pattern found
            } else if (i < text.length() && pattern.charAt(j) != text.charAt(i)) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return -1; // pattern not found
    }
    
    private int[] computeLPS(String pattern) {
        int[] lps = new int[pattern.length()];
        int len = 0;
        int i = 1;
        
        while (i < pattern.length()) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }
}
```

**Time Complexity:** O(n + m)
**Space Complexity:** O(m)

## 6. Segment Tree

**Problem:** Implement a Segment Tree for range sum queries.

**Solution:**
```java
class SegmentTree {
    private int[] tree;
    private int n;
    
    public SegmentTree(int[] nums) {
        n = nums.length;
        tree = new int[4 * n];
        build(nums, 0, 0, n - 1);
    }
    
    private void build(int[] nums, int node, int start, int end) {
        if (start == end) {
            tree[node] = nums[start];
            return;
        }
        
        int mid = start + (end - start) / 2;
        build(nums, 2 * node + 1, start, mid);
        build(nums, 2 * node + 2, mid + 1, end);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    
    public void update(int index, int val) {
        update(0, 0, n - 1, index, val);
    }
    
    private void update(int node, int start, int end, int index, int val) {
        if (start == end) {
            tree[node] = val;
            return;
        }
        
        int mid = start + (end - start) / 2;
        if (index <= mid) {
            update(2 * node + 1, start, mid, index, val);
        } else {
            update(2 * node + 2, mid + 1, end, index, val);
        }
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    
    public int query(int left, int right) {
        return query(0, 0, n - 1, left, right);
    }
    
    private int query(int node, int start, int end, int left, int right) {
        if (start > right || end < left) return 0;
        if (start >= left && end <= right) return tree[node];
        
        int mid = start + (end - start) / 2;
        return query(2 * node + 1, start, mid, left, right) +
               query(2 * node + 2, mid + 1, end, left, right);
    }
}
```

**Time Complexity:**
- Build: O(n)
- Update: O(log n)
- Query: O(log n)
**Space Complexity:** O(n)

## 7. Fenwick Tree (Binary Indexed Tree)

**Problem:** Implement a Fenwick Tree for prefix sum queries.

**Solution:**
```java
class FenwickTree {
    private int[] tree;
    private int n;
    
    public FenwickTree(int n) {
        this.n = n;
        tree = new int[n + 1];
    }
    
    public void update(int index, int val) {
        index++;
        while (index <= n) {
            tree[index] += val;
            index += index & (-index);
        }
    }
    
    public int query(int index) {
        index++;
        int sum = 0;
        while (index > 0) {
            sum += tree[index];
            index -= index & (-index);
        }
        return sum;
    }
    
    public int queryRange(int left, int right) {
        return query(right) - query(left - 1);
    }
}
```

**Time Complexity:**
- Update: O(log n)
- Query: O(log n)
**Space Complexity:** O(n)

## 8. 2D Fenwick Tree

**Problem:** Implement a 2D Fenwick Tree for 2D range sum queries.

**Solution:**
```java
class FenwickTree2D {
    private int[][] tree;
    private int rows;
    private int cols;
    
    public FenwickTree2D(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        tree = new int[rows + 1][cols + 1];
    }
    
    public void update(int row, int col, int val) {
        row++;
        col++;
        while (row <= rows) {
            int col1 = col;
            while (col1 <= cols) {
                tree[row][col1] += val;
                col1 += col1 & (-col1);
            }
            row += row & (-row);
        }
    }
    
    public int query(int row, int col) {
        row++;
        col++;
        int sum = 0;
        while (row > 0) {
            int col1 = col;
            while (col1 > 0) {
                sum += tree[row][col1];
                col1 -= col1 & (-col1);
            }
            row -= row & (-row);
        }
        return sum;
    }
    
    public int queryRange(int row1, int col1, int row2, int col2) {
        return query(row2, col2) - query(row2, col1 - 1) -
               query(row1 - 1, col2) + query(row1 - 1, col1 - 1);
    }
}
```

**Time Complexity:**
- Update: O(log m * log n)
- Query: O(log m * log n)
**Space Complexity:** O(m * n)

## Key Takeaways

1. These data structures and algorithms are perfect for:
   - Efficient range queries
   - Graph algorithms
   - String matching
   - Dynamic programming optimization
   - Advanced tree operations

2. Common patterns:
   - Path compression and union by rank
   - Priority queue for shortest paths
   - Prefix sums and range queries
   - String pattern matching
   - Tree-based optimizations

3. Tips:
   - Choose the right data structure for the problem
   - Consider space-time tradeoffs
   - Handle edge cases carefully
   - Optimize for specific operations
   - Understand the underlying principles 