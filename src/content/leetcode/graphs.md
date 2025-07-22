---
title: Graphs
description: Java solutions with explanations, time and space complexity for Graph problems.
date: "June 1 2025"
---

# Graph Pattern

Graphs are a fundamental data structure used to represent relationships between objects. They're particularly useful for:
- Modeling networks and connections
- Path finding and traversal
- Cycle detection
- Topological sorting
- Connected components analysis

## 1. Number of Islands (Medium)

**Problem:** Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

**Solution:**
```java
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }
    
    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length 
            || grid[i][j] == '0') {
            return;
        }
        
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
}
```

**Time Complexity:** O(m * n) where m,n are dimensions of grid
**Space Complexity:** O(m * n) for recursion stack

## 2. Clone Graph (Medium)

**Problem:** Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

**Solution:**
```java
class Solution {
    private Map<Node, Node> visited = new HashMap<>();
    
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        
        if (visited.containsKey(node)) {
            return visited.get(node);
        }
        
        Node clone = new Node(node.val, new ArrayList<>());
        visited.put(node, clone);
        
        for (Node neighbor : node.neighbors) {
            clone.neighbors.add(cloneGraph(neighbor));
        }
        
        return clone;
    }
}
```

**Time Complexity:** O(V + E) where V is vertices and E is edges
**Space Complexity:** O(V) for the hash map

## 3. Max Area of Island (Medium)

**Problem:** You are given an m x n binary matrix grid. An island is a group of 1's connected 4-directionally. Return the maximum area of an island in grid.

**Solution:**
```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int maxArea = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    maxArea = Math.max(maxArea, dfs(grid, i, j));
                }
            }
        }
        return maxArea;
    }
    
    private int dfs(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length 
            || grid[i][j] == 0) {
            return 0;
        }
        
        grid[i][j] = 0;
        return 1 + dfs(grid, i + 1, j) + dfs(grid, i - 1, j) 
                 + dfs(grid, i, j + 1) + dfs(grid, i, j - 1);
    }
}
```

**Time Complexity:** O(m * n) where m,n are dimensions of grid
**Space Complexity:** O(m * n) for recursion stack

## 4. Pacific Atlantic Water Flow (Medium)

**Problem:** Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, find the coordinates of cells that can flow to both the Pacific and Atlantic oceans.

**Solution:**
```java
class Solution {
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> result = new ArrayList<>();
        if (heights == null || heights.length == 0) return result;
        
        int m = heights.length, n = heights[0].length;
        boolean[][] pacific = new boolean[m][n];
        boolean[][] atlantic = new boolean[m][n];
        
        for (int i = 0; i < m; i++) {
            dfs(heights, pacific, i, 0, Integer.MIN_VALUE);
            dfs(heights, atlantic, i, n - 1, Integer.MIN_VALUE);
        }
        
        for (int j = 0; j < n; j++) {
            dfs(heights, pacific, 0, j, Integer.MIN_VALUE);
            dfs(heights, atlantic, m - 1, j, Integer.MIN_VALUE);
        }
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacific[i][j] && atlantic[i][j]) {
                    result.add(Arrays.asList(i, j));
                }
            }
        }
        
        return result;
    }
    
    private void dfs(int[][] heights, boolean[][] visited, int i, int j, int prev) {
        if (i < 0 || i >= heights.length || j < 0 || j >= heights[0].length 
            || visited[i][j] || heights[i][j] < prev) {
            return;
        }
        
        visited[i][j] = true;
        dfs(heights, visited, i + 1, j, heights[i][j]);
        dfs(heights, visited, i - 1, j, heights[i][j]);
        dfs(heights, visited, i, j + 1, heights[i][j]);
        dfs(heights, visited, i, j - 1, heights[i][j]);
    }
}
```

**Time Complexity:** O(m * n) where m,n are dimensions of matrix
**Space Complexity:** O(m * n) for visited arrays

## 5. Surrounded Regions (Medium)

**Problem:** Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.

**Solution:**
```java
class Solution {
    public void solve(char[][] board) {
        if (board == null || board.length == 0) return;
        
        int m = board.length, n = board[0].length;
        
        // Mark 'O's on the border and their connected 'O's
        for (int i = 0; i < m; i++) {
            dfs(board, i, 0);
            dfs(board, i, n - 1);
        }
        for (int j = 0; j < n; j++) {
            dfs(board, 0, j);
            dfs(board, m - 1, j);
        }
        
        // Convert remaining 'O's to 'X's and '#'s back to 'O's
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == '#') {
                    board[i][j] = 'O';
                }
            }
        }
    }
    
    private void dfs(char[][] board, int i, int j) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length 
            || board[i][j] != 'O') {
            return;
        }
        
        board[i][j] = '#';
        dfs(board, i + 1, j);
        dfs(board, i - 1, j);
        dfs(board, i, j + 1);
        dfs(board, i, j - 1);
    }
}
```

**Time Complexity:** O(m * n) where m,n are dimensions of board
**Space Complexity:** O(m * n) for recursion stack

## 6. Rotting Oranges (Medium)

**Problem:** In a given grid, each cell can have one of three values: 0 representing an empty cell, 1 representing a fresh orange, or 2 representing a rotten orange. Return the minimum number of minutes that must elapse until no cell has a fresh orange.

**Solution:**
```java
class Solution {
    public int orangesRotting(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        Queue<int[]> queue = new LinkedList<>();
        int fresh = 0;
        
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j});
                } else if (grid[i][j] == 1) {
                    fresh++;
                }
            }
        }
        
        if (fresh == 0) return 0;
        
        int minutes = 0;
        int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        while (!queue.isEmpty() && fresh > 0) {
            int size = queue.size();
            minutes++;
            
            for (int i = 0; i < size; i++) {
                int[] point = queue.poll();
                
                for (int[] dir : directions) {
                    int x = point[0] + dir[0];
                    int y = point[1] + dir[1];
                    
                    if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length 
                        && grid[x][y] == 1) {
                        grid[x][y] = 2;
                        queue.offer(new int[]{x, y});
                        fresh--;
                    }
                }
            }
        }
        
        return fresh == 0 ? minutes : -1;
    }
}
```

**Time Complexity:** O(m * n) where m,n are dimensions of grid
**Space Complexity:** O(m * n) for the queue

## 7. Walls and Gates (Medium)

**Problem:** You are given an m x n grid rooms initialized with these three possible values: -1 (wall), 0 (gate), INF (empty room). Fill each empty room with the distance to its nearest gate.

**Solution:**
```java
class Solution {
    public void wallsAndGates(int[][] rooms) {
        if (rooms == null || rooms.length == 0) return;
        
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < rooms.length; i++) {
            for (int j = 0; j < rooms[0].length; j++) {
                if (rooms[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                }
            }
        }
        
        int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int row = point[0], col = point[1];
            
            for (int[] dir : directions) {
                int r = row + dir[0];
                int c = col + dir[1];
                
                if (r >= 0 && r < rooms.length && c >= 0 && c < rooms[0].length 
                    && rooms[r][c] == Integer.MAX_VALUE) {
                    rooms[r][c] = rooms[row][col] + 1;
                    queue.offer(new int[]{r, c});
                }
            }
        }
    }
}
```

**Time Complexity:** O(m * n) where m,n are dimensions of rooms
**Space Complexity:** O(m * n) for the queue

## 8. Course Schedule (Medium)

**Problem:** There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. Return true if you can finish all courses.

**Solution:**
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] prerequisite : prerequisites) {
            graph.get(prerequisite[1]).add(prerequisite[0]);
        }
        
        boolean[] visited = new boolean[numCourses];
        boolean[] recursionStack = new boolean[numCourses];
        
        for (int i = 0; i < numCourses; i++) {
            if (hasCycle(graph, i, visited, recursionStack)) {
                return false;
            }
        }
        
        return true;
    }
    
    private boolean hasCycle(List<List<Integer>> graph, int course, 
                           boolean[] visited, boolean[] recursionStack) {
        if (recursionStack[course]) return true;
        if (visited[course]) return false;
        
        visited[course] = true;
        recursionStack[course] = true;
        
        for (int neighbor : graph.get(course)) {
            if (hasCycle(graph, neighbor, visited, recursionStack)) {
                return true;
            }
        }
        
        recursionStack[course] = false;
        return false;
    }
}
```

**Time Complexity:** O(V + E) where V is vertices and E is edges
**Space Complexity:** O(V) for visited arrays

## 9. Course Schedule II (Medium)

**Problem:** Return the ordering of courses you should take to finish all courses.

**Solution:**
```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] prerequisite : prerequisites) {
            graph.get(prerequisite[1]).add(prerequisite[0]);
        }
        
        boolean[] visited = new boolean[numCourses];
        boolean[] recursionStack = new boolean[numCourses];
        List<Integer> order = new ArrayList<>();
        
        for (int i = 0; i < numCourses; i++) {
            if (hasCycle(graph, i, visited, recursionStack, order)) {
                return new int[0];
            }
        }
        
        Collections.reverse(order);
        return order.stream().mapToInt(Integer::intValue).toArray();
    }
    
    private boolean hasCycle(List<List<Integer>> graph, int course, 
                           boolean[] visited, boolean[] recursionStack, 
                           List<Integer> order) {
        if (recursionStack[course]) return true;
        if (visited[course]) return false;
        
        visited[course] = true;
        recursionStack[course] = true;
        
        for (int neighbor : graph.get(course)) {
            if (hasCycle(graph, neighbor, visited, recursionStack, order)) {
                return true;
            }
        }
        
        recursionStack[course] = false;
        order.add(course);
        return false;
    }
}
```

**Time Complexity:** O(V + E) where V is vertices and E is edges
**Space Complexity:** O(V) for visited arrays and order list

## 10. Redundant Connection (Medium)

**Problem:** Given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. Return an edge that can be removed so that the resulting graph is a tree of n nodes.

**Solution:**
```java
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        int[] parent = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            parent[i] = i;
        }
        
        for (int[] edge : edges) {
            int x = find(parent, edge[0]);
            int y = find(parent, edge[1]);
            
            if (x == y) {
                return edge;
            }
            
            parent[y] = x;
        }
        
        return new int[0];
    }
    
    private int find(int[] parent, int x) {
        if (parent[x] != x) {
            parent[x] = find(parent, parent[x]);
        }
        return parent[x];
    }
}
```

**Time Complexity:** O(n) where n is the number of edges
**Space Complexity:** O(n) for the parent array

## 11. Number of Connected Components in an Undirected Graph (Medium)

**Problem:** Given n nodes labeled from 0 to n - 1 and a list of undirected edges, write a function to find the number of connected components in an undirected graph.

**Solution:**
```java
class Solution {
    public int countComponents(int n, int[][] edges) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        
        for (int[] edge : edges) {
            int x = find(parent, edge[0]);
            int y = find(parent, edge[1]);
            
            if (x != y) {
                parent[y] = x;
                n--;
            }
        }
        
        return n;
    }
    
    private int find(int[] parent, int x) {
        if (parent[x] != x) {
            parent[x] = find(parent, parent[x]);
        }
        return parent[x];
    }
}
```

**Time Complexity:** O(E * α(n)) where E is number of edges and α is inverse Ackermann function
**Space Complexity:** O(n) for the parent array

## 12. Graph Valid Tree (Medium)

**Problem:** Given n nodes labeled from 0 to n-1 and a list of undirected edges, write a function to check whether these edges make up a valid tree.

**Solution:**
```java
class Solution {
    public boolean validTree(int n, int[][] edges) {
        if (edges.length != n - 1) return false;
        
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        
        boolean[] visited = new boolean[n];
        if (hasCycle(graph, 0, -1, visited)) {
            return false;
        }
        
        for (boolean v : visited) {
            if (!v) return false;
        }
        
        return true;
    }
    
    private boolean hasCycle(List<List<Integer>> graph, int node, int parent, 
                           boolean[] visited) {
        visited[node] = true;
        
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                if (hasCycle(graph, neighbor, node, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        
        return false;
    }
}
```

**Time Complexity:** O(V + E) where V is vertices and E is edges
**Space Complexity:** O(V) for visited array

## 13. Word Ladder (Hard)

**Problem:** Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord.

**Solution:**
```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> set = new HashSet<>(wordList);
        if (!set.contains(endWord)) return 0;
        
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);
        set.remove(beginWord);
        
        int level = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            
            for (int i = 0; i < size; i++) {
                String current = queue.poll();
                
                if (current.equals(endWord)) {
                    return level;
                }
                
                char[] chars = current.toCharArray();
                for (int j = 0; j < chars.length; j++) {
                    char original = chars[j];
                    
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == original) continue;
                        
                        chars[j] = c;
                        String newWord = new String(chars);
                        
                        if (set.contains(newWord)) {
                            queue.offer(newWord);
                            set.remove(newWord);
                        }
                    }
                    
                    chars[j] = original;
                }
            }
            
            level++;
        }
        
        return 0;
    }
}
```

**Time Complexity:** O(26 * L * N) where L is word length and N is wordList size
**Space Complexity:** O(N) for the set and queue

## Key Takeaways

1. Graph problems often involve:
   - DFS/BFS traversal
   - Cycle detection
   - Connected components
   - Topological sorting
   - Shortest path finding

2. Common patterns:
   - Adjacency list/matrix representation
   - Visited arrays/maps
   - Queue for BFS
   - Stack/Recursion for DFS
   - Union-Find for connectivity

3. Tips:
   - Choose between DFS and BFS based on problem
   - Consider using visited sets for cycles
   - Use appropriate data structures
   - Handle edge cases carefully
   - Consider space-time tradeoffs 