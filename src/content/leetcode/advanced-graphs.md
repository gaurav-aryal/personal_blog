---
title: Advanced Graphs
description: Java solutions with explanations, time and space complexity for Advanced Graph problems.
date: "June 1 2025"
---

# Advanced Graph Pattern

Advanced Graph algorithms extend basic graph concepts to solve complex problems involving:
- Eulerian paths and circuits
- Minimum Spanning Trees (MST)
- Shortest Path algorithms
- Topological sorting with constraints
- Network flow optimization
- Multi-source shortest paths

## 1. Reconstruct Itinerary (Hard)

**Problem:** Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order.

**Solution:**
```java
class Solution {
    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String, PriorityQueue<String>> graph = new HashMap<>();
        List<String> result = new ArrayList<>();
        
        // Build graph
        for (List<String> ticket : tickets) {
            graph.putIfAbsent(ticket.get(0), new PriorityQueue<>());
            graph.get(ticket.get(0)).offer(ticket.get(1));
        }
        
        dfs("JFK", graph, result);
        Collections.reverse(result);
        return result;
    }
    
    private void dfs(String airport, Map<String, PriorityQueue<String>> graph, 
                    List<String> result) {
        PriorityQueue<String> destinations = graph.get(airport);
        
        while (destinations != null && !destinations.isEmpty()) {
            String next = destinations.poll();
            dfs(next, graph, result);
        }
        
        result.add(airport);
    }
}
```

**Time Complexity:** O(E log E) where E is the number of edges
**Space Complexity:** O(E) for the graph and result

## 2. Min Cost to Connect All Points (Medium)

**Problem:** You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi]. Return the minimum cost to make all points connected.

**Solution:**
```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        boolean[] visited = new boolean[n];
        int result = 0;
        int edges = 0;
        
        // Add all edges from first point
        for (int i = 1; i < n; i++) {
            pq.offer(new int[]{0, i, 
                Math.abs(points[0][0] - points[i][0]) + 
                Math.abs(points[0][1] - points[i][1])});
        }
        visited[0] = true;
        
        while (!pq.isEmpty() && edges < n - 1) {
            int[] edge = pq.poll();
            int point2 = edge[1];
            
            if (visited[point2]) continue;
            
            visited[point2] = true;
            result += edge[2];
            edges++;
            
            // Add new edges from the newly connected point
            for (int i = 0; i < n; i++) {
                if (!visited[i]) {
                    pq.offer(new int[]{point2, i, 
                        Math.abs(points[point2][0] - points[i][0]) + 
                        Math.abs(points[point2][1] - points[i][1])});
                }
            }
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n² log n) where n is the number of points
**Space Complexity:** O(n²) for the priority queue

## 3. Network Delay Time (Medium)

**Problem:** You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

**Solution:**
```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] time : times) {
            graph.putIfAbsent(time[0], new ArrayList<>());
            graph.get(time[0]).add(new int[]{time[1], time[2]});
        }
        
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        pq.offer(new int[]{k, 0});
        
        Map<Integer, Integer> dist = new HashMap<>();
        
        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int node = curr[0], time = curr[1];
            
            if (dist.containsKey(node)) continue;
            dist.put(node, time);
            
            if (graph.containsKey(node)) {
                for (int[] edge : graph.get(node)) {
                    int neighbor = edge[0], weight = edge[1];
                    if (!dist.containsKey(neighbor)) {
                        pq.offer(new int[]{neighbor, time + weight});
                    }
                }
            }
        }
        
        if (dist.size() != n) return -1;
        return Collections.max(dist.values());
    }
}
```

**Time Complexity:** O(E log V) where E is edges and V is vertices
**Space Complexity:** O(V + E) for the graph and distance map

## 4. Swim in Rising Water (Hard)

**Problem:** On an N x N grid, each square grid[i][j] represents the elevation at that point (i,j). Now rain starts to fall. At time t, the depth of the water everywhere is t. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most t.

**Solution:**
```java
class Solution {
    public int swimInWater(int[][] grid) {
        int n = grid.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        boolean[][] visited = new boolean[n][n];
        
        pq.offer(new int[]{0, 0, grid[0][0]});
        visited[0][0] = true;
        
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        
        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int i = curr[0], j = curr[1], time = curr[2];
            
            if (i == n - 1 && j == n - 1) return time;
            
            for (int[] dir : directions) {
                int ni = i + dir[0], nj = j + dir[1];
                
                if (ni >= 0 && ni < n && nj >= 0 && nj < n && !visited[ni][nj]) {
                    visited[ni][nj] = true;
                    pq.offer(new int[]{ni, nj, Math.max(time, grid[ni][nj])});
                }
            }
        }
        
        return -1;
    }
}
```

**Time Complexity:** O(n² log n) where n is the grid size
**Space Complexity:** O(n²) for visited array and priority queue

## 5. Alien Dictionary (Hard)

**Problem:** There is a new alien language that uses the English alphabet. However, the order among letters is unknown to you. You are given a list of strings words from the dictionary, where words are sorted lexicographically by the rules of this new language.

**Solution:**
```java
class Solution {
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> graph = new HashMap<>();
        Map<Character, Integer> inDegree = new HashMap<>();
        
        // Initialize inDegree for all characters
        for (String word : words) {
            for (char c : word.toCharArray()) {
                inDegree.put(c, 0);
            }
        }
        
        // Build graph
        for (int i = 0; i < words.length - 1; i++) {
            String word1 = words[i], word2 = words[i + 1];
            
            if (word1.length() > word2.length() && word1.startsWith(word2)) {
                return "";
            }
            
            for (int j = 0; j < Math.min(word1.length(), word2.length()); j++) {
                char c1 = word1.charAt(j), c2 = word2.charAt(j);
                
                if (c1 != c2) {
                    graph.putIfAbsent(c1, new HashSet<>());
                    if (!graph.get(c1).contains(c2)) {
                        graph.get(c1).add(c2);
                        inDegree.put(c2, inDegree.get(c2) + 1);
                    }
                    break;
                }
            }
        }
        
        // Topological sort
        StringBuilder result = new StringBuilder();
        Queue<Character> queue = new LinkedList<>();
        
        for (char c : inDegree.keySet()) {
            if (inDegree.get(c) == 0) {
                queue.offer(c);
            }
        }
        
        while (!queue.isEmpty()) {
            char c = queue.poll();
            result.append(c);
            
            if (graph.containsKey(c)) {
                for (char neighbor : graph.get(c)) {
                    inDegree.put(neighbor, inDegree.get(neighbor) - 1);
                    if (inDegree.get(neighbor) == 0) {
                        queue.offer(neighbor);
                    }
                }
            }
        }
        
        return result.length() == inDegree.size() ? result.toString() : "";
    }
}
```

**Time Complexity:** O(C) where C is the total number of characters
**Space Complexity:** O(1) as there are only 26 characters

## 6. Cheapest Flights Within K Stops (Medium)

**Problem:** There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.

**Solution:**
```java
class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] flight : flights) {
            graph.putIfAbsent(flight[0], new ArrayList<>());
            graph.get(flight[0]).add(new int[]{flight[1], flight[2]});
        }
        
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        pq.offer(new int[]{0, src, k + 1});
        
        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int price = curr[0], city = curr[1], stops = curr[2];
            
            if (city == dst) return price;
            if (stops == 0) continue;
            
            if (graph.containsKey(city)) {
                for (int[] flight : graph.get(city)) {
                    pq.offer(new int[]{price + flight[1], flight[0], stops - 1});
                }
            }
        }
        
        return -1;
    }
}
```

**Time Complexity:** O(E * K * log(E*K)) where E is number of flights and K is number of stops
**Space Complexity:** O(E * K) for the priority queue

## Key Takeaways

1. Advanced Graph algorithms are used for:
   - Finding optimal paths
   - Network optimization
   - Resource allocation
   - Scheduling problems
   - Constraint satisfaction

2. Common patterns:
   - Dijkstra's algorithm for shortest paths
   - Kruskal's/Prim's for MST
   - Topological sorting
   - Eulerian paths
   - Multi-source BFS

3. Tips:
   - Choose appropriate algorithm based on constraints
   - Consider using priority queues for optimization
   - Handle cycles and negative weights carefully
   - Use appropriate data structures for efficiency
   - Consider both time and space complexity 