---
title: Heap / Priority Queue
description: Java solutions with explanations, time and space complexity for Heap/Priority Queue problems from Blind 75.
date: "June 1 2025"
order: 8
---

# Heap / Priority Queue

This section covers problems that can be efficiently solved using heaps and priority queues.

## 1. Find Median from Data Stream (Hard)

**Problem:** The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

Implement the MedianFinder class:
- `MedianFinder()` initializes the MedianFinder object.
- `addNum(int num)` adds the integer num from the data stream to the data structure.
- `findMedian()` returns the median of all elements so far.

**Example:**
```
Input: ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output: [null, null, null, 1.5, null, 2.0]
```

**Solution:**
```java
class MedianFinder {
    private PriorityQueue<Integer> maxHeap; // stores the smaller half
    private PriorityQueue<Integer> minHeap; // stores the larger half
    
    public MedianFinder() {
        maxHeap = new PriorityQueue<>((a, b) -> b - a);
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

**Time Complexity:** O(log n) for addNum, O(1) for findMedian
**Space Complexity:** O(n)

---

## 2. Merge K Sorted Lists (Hard)

**Problem:** You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

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
        
        // Add first node of each list to priority queue
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

## 3. Top K Frequent Elements (Medium)

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
        Map<Integer, Integer> frequency = new HashMap<>();
        
        for (int num : nums) {
            frequency.put(num, frequency.getOrDefault(num, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> pq = 
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        
        for (Map.Entry<Integer, Integer> entry : frequency.entrySet()) {
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

## Key Takeaways

1. **Two Heaps**: Use min and max heaps for median problems
2. **K-th Element**: Use heap to find k-th largest/smallest elements
3. **Frequency**: Combine hash maps with heaps for frequency problems
4. **Merge Operations**: Use heaps for merging multiple sorted structures 