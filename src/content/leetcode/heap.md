---
title: Heap / PriorityQueue
description: Java solutions with explanations, time and space complexity for Heap / PriorityQueue problems.
date: "June 1 2025"
---

# Heap/PriorityQueue Pattern

Heaps (Priority Queues) are specialized tree-based data structures that satisfy the heap property. They're particularly useful for:
- Finding kth largest/smallest elements
- Implementing priority-based scheduling
- Merge k sorted lists
- Finding running medians
- Task scheduling

## 1. Kth Largest Element in a Stream (Easy)

**Problem:** Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.

**Solution:**
```java
class KthLargest {
    private PriorityQueue<Integer> minHeap;
    private int k;
    
    public KthLargest(int k, int[] nums) {
        this.k = k;
        minHeap = new PriorityQueue<>();
        
        for (int num : nums) {
            add(num);
        }
    }
    
    public int add(int val) {
        minHeap.offer(val);
        if (minHeap.size() > k) {
            minHeap.poll();
        }
        return minHeap.peek();
    }
}
```

**Time Complexity:**
- Constructor: O(n log k) where n is the length of nums
- add: O(log k)
**Space Complexity:** O(k) for the heap

## 2. Last Stone Weight (Easy)

**Problem:** We have a collection of stones, each stone has a positive integer weight. Each turn, we choose the two heaviest stones and smash them together. Return the weight of the last remaining stone.

**Solution:**
```java
class Solution {
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        
        for (int stone : stones) {
            maxHeap.offer(stone);
        }
        
        while (maxHeap.size() > 1) {
            int stone1 = maxHeap.poll();
            int stone2 = maxHeap.poll();
            
            if (stone1 != stone2) {
                maxHeap.offer(stone1 - stone2);
            }
        }
        
        return maxHeap.isEmpty() ? 0 : maxHeap.poll();
    }
}
```

**Time Complexity:** O(n log n) where n is the number of stones
**Space Complexity:** O(n) for the heap

## 3. K Closest Points to Origin (Medium)

**Problem:** Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

**Solution:**
```java
class Solution {
    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
            (a, b) -> (b[0] * b[0] + b[1] * b[1]) - (a[0] * a[0] + a[1] * a[1])
        );
        
        for (int[] point : points) {
            maxHeap.offer(point);
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }
        
        int[][] result = new int[k][2];
        for (int i = 0; i < k; i++) {
            result[i] = maxHeap.poll();
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n log k) where n is the number of points
**Space Complexity:** O(k) for the heap

## 4. Kth Largest Element in an Array (Medium)

**Problem:** Given an integer array nums and an integer k, return the kth largest element in the array.

**Solution:**
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        for (int num : nums) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        
        return minHeap.peek();
    }
}
```

**Time Complexity:** O(n log k) where n is the length of nums
**Space Complexity:** O(k) for the heap

## 5. Task Scheduler (Medium)

**Problem:** Given a characters array tasks, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle. Return the least number of units of times that the CPU will take to finish all the given tasks.

**Solution:**
```java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        int[] frequencies = new int[26];
        for (char task : tasks) {
            frequencies[task - 'A']++;
        }
        
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        for (int freq : frequencies) {
            if (freq > 0) {
                maxHeap.offer(freq);
            }
        }
        
        int time = 0;
        while (!maxHeap.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            int cycle = n + 1;
            
            while (cycle > 0 && !maxHeap.isEmpty()) {
                int freq = maxHeap.poll();
                if (freq > 1) {
                    temp.add(freq - 1);
                }
                cycle--;
                time++;
            }
            
            for (int freq : temp) {
                maxHeap.offer(freq);
            }
            
            if (!maxHeap.isEmpty()) {
                time += cycle;
            }
        }
        
        return time;
    }
}
```

**Time Complexity:** O(n log 26) where n is the number of tasks
**Space Complexity:** O(1) as we only store 26 characters

## 6. Design Twitter (Medium)

**Problem:** Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the 10 most recent tweets in the user's news feed.

**Solution:**
```java
class Twitter {
    private static int timeStamp = 0;
    private Map<Integer, User> userMap;
    
    private class Tweet {
        int id;
        int time;
        Tweet next;
        
        public Tweet(int id) {
            this.id = id;
            this.time = timeStamp++;
            this.next = null;
        }
    }
    
    private class User {
        int id;
        Set<Integer> following;
        Tweet tweetHead;
        
        public User(int id) {
            this.id = id;
            following = new HashSet<>();
            following.add(id);
            tweetHead = null;
        }
        
        public void follow(int id) {
            following.add(id);
        }
        
        public void unfollow(int id) {
            following.remove(id);
        }
        
        public void post(int id) {
            Tweet t = new Tweet(id);
            t.next = tweetHead;
            tweetHead = t;
        }
    }
    
    public Twitter() {
        userMap = new HashMap<>();
    }
    
    public void postTweet(int userId, int tweetId) {
        if (!userMap.containsKey(userId)) {
            userMap.put(userId, new User(userId));
        }
        userMap.get(userId).post(tweetId);
    }
    
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        if (!userMap.containsKey(userId)) return res;
        
        Set<Integer> users = userMap.get(userId).following;
        PriorityQueue<Tweet> pq = new PriorityQueue<>((a, b) -> b.time - a.time);
        
        for (int user : users) {
            Tweet t = userMap.get(user).tweetHead;
            if (t != null) {
                pq.offer(t);
            }
        }
        
        int count = 0;
        while (!pq.isEmpty() && count < 10) {
            Tweet t = pq.poll();
            res.add(t.id);
            count++;
            if (t.next != null) {
                pq.offer(t.next);
            }
        }
        
        return res;
    }
    
    public void follow(int followerId, int followeeId) {
        if (!userMap.containsKey(followerId)) {
            userMap.put(followerId, new User(followerId));
        }
        if (!userMap.containsKey(followeeId)) {
            userMap.put(followeeId, new User(followeeId));
        }
        userMap.get(followerId).follow(followeeId);
    }
    
    public void unfollow(int followerId, int followeeId) {
        if (!userMap.containsKey(followerId) || followerId == followeeId) return;
        userMap.get(followerId).unfollow(followeeId);
    }
}
```

**Time Complexity:**
- postTweet: O(1)
- getNewsFeed: O(n log k) where n is number of tweets and k is 10
- follow/unfollow: O(1)
**Space Complexity:** O(n) where n is the number of users and tweets

## 7. Find Median from Data Stream (Hard)

**Problem:** The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

**Solution:**
```java
class MedianFinder {
    private PriorityQueue<Integer> maxHeap; // lower half
    private PriorityQueue<Integer> minHeap; // upper half
    
    public MedianFinder() {
        maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        minHeap = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());
        
        if (maxHeap.size() < minHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }
    
    public double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.peek();
        }
        return (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
}
```

**Time Complexity:**
- addNum: O(log n)
- findMedian: O(1)
**Space Complexity:** O(n) where n is the number of elements

## Key Takeaways

1. Heap/PriorityQueue is perfect for:
   - Finding kth largest/smallest elements
   - Implementing priority-based scheduling
   - Merge k sorted lists
   - Finding running medians
   - Task scheduling

2. Common patterns:
   - Min heap for kth largest
   - Max heap for kth smallest
   - Two heaps for median
   - Custom comparators
   - Size management

3. Tips:
   - Choose between min and max heap based on problem
   - Consider using two heaps for complex problems
   - Maintain heap size for efficiency
   - Use custom comparators when needed
   - Consider space-time tradeoffs 