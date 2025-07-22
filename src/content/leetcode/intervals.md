---
title: Intervals
description: Java solutions with explanations, time and space complexity for Intervals problems.
date: "June 1 2025"
---

# Intervals Pattern

Interval problems deal with ranges or periods of time/values and often require:
- Sorting intervals
- Merging overlapping intervals
- Finding non-overlapping intervals
- Scheduling conflicts
- Resource allocation

## 1. Insert Interval (Medium)

**Problem:** You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

**Solution:**
```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> result = new ArrayList<>();
        int i = 0;
        
        // Add all intervals before newInterval
        while (i < intervals.length && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i++]);
        }
        
        // Merge overlapping intervals
        while (i < intervals.length && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.add(newInterval);
        
        // Add remaining intervals
        while (i < intervals.length) {
            result.add(intervals[i++]);
        }
        
        return result.toArray(new int[result.size()][]);
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

## 2. Merge Intervals (Medium)

**Problem:** Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

**Solution:**
```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        
        // Sort intervals by start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> result = new ArrayList<>();
        int[] current = intervals[0];
        result.add(current);
        
        for (int[] interval : intervals) {
            if (interval[0] <= current[1]) {
                // Overlapping intervals, update the end
                current[1] = Math.max(current[1], interval[1]);
            } else {
                // Non-overlapping interval, add to result
                current = interval;
                result.add(current);
            }
        }
        
        return result.toArray(new int[result.size()][]);
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

## 3. Non-overlapping Intervals (Medium)

**Problem:** Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

**Solution:**
```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        // Sort intervals by end time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));
        
        int count = 0;
        int end = intervals[0][1];
        
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= end) {
                // Non-overlapping interval
                end = intervals[i][1];
            } else {
                // Overlapping interval, increment count
                count++;
            }
        }
        
        return count;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(1)

## 4. Meeting Rooms (Easy)

**Problem:** Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.

**Solution:**
```java
class Solution {
    public boolean canAttendMeetings(int[][] intervals) {
        // Sort intervals by start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] < intervals[i - 1][1]) {
                return false;
            }
        }
        
        return true;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(1)

## 5. Meeting Rooms II (Medium)

**Problem:** Given an array of meeting time intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

**Solution:**
```java
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        // Sort start times and end times
        int[] starts = new int[intervals.length];
        int[] ends = new int[intervals.length];
        
        for (int i = 0; i < intervals.length; i++) {
            starts[i] = intervals[i][0];
            ends[i] = intervals[i][1];
        }
        
        Arrays.sort(starts);
        Arrays.sort(ends);
        
        int rooms = 0;
        int endIndex = 0;
        
        for (int start : starts) {
            if (start < ends[endIndex]) {
                // Need a new room
                rooms++;
            } else {
                // Reuse a room
                endIndex++;
            }
        }
        
        return rooms;
    }
}
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)

## 6. Minimum Interval to Include Each Query (Hard)

**Problem:** You are given a 2D integer array intervals, where intervals[i] = [lefti, righti] describes the ith interval starting at lefti and ending at righti (inclusive). You are also given an integer array queries. The answer to the jth query is the size of the smallest interval i such that lefti <= queries[j] <= righti.

**Solution:**
```java
class Solution {
    public int[] minInterval(int[][] intervals, int[] queries) {
        // Sort intervals by start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        // Create array of queries with their original indices
        int[][] queriesWithIndex = new int[queries.length][2];
        for (int i = 0; i < queries.length; i++) {
            queriesWithIndex[i] = new int[]{queries[i], i};
        }
        Arrays.sort(queriesWithIndex, (a, b) -> Integer.compare(a[0], b[0]));
        
        int[] result = new int[queries.length];
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        int intervalIndex = 0;
        
        for (int[] query : queriesWithIndex) {
            int queryVal = query[0];
            int queryIndex = query[1];
            
            // Add all intervals that start before or at the query value
            while (intervalIndex < intervals.length && intervals[intervalIndex][0] <= queryVal) {
                int[] interval = intervals[intervalIndex];
                int size = interval[1] - interval[0] + 1;
                pq.offer(new int[]{size, interval[1]});
                intervalIndex++;
            }
            
            // Remove intervals that end before the query value
            while (!pq.isEmpty() && pq.peek()[1] < queryVal) {
                pq.poll();
            }
            
            result[queryIndex] = pq.isEmpty() ? -1 : pq.peek()[0];
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n log n + q log q)
**Space Complexity:** O(n + q)

## Key Takeaways

1. Interval problems often require:
   - Sorting intervals by start or end time
   - Tracking overlapping intervals
   - Managing multiple intervals simultaneously
   - Handling edge cases

2. Common patterns:
   - Sort and merge
   - Two-pointer technique
   - Priority queue for scheduling
   - Line sweep algorithm
   - Interval tree for range queries

3. Tips:
   - Consider sorting by start or end time
   - Use appropriate data structures
   - Handle edge cases carefully
   - Consider space-time tradeoffs
   - Think about interval relationships 