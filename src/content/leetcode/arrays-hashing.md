---
title: Arrays & Hashing
description: Java solutions with explanations, time and space complexity for Arrays & Hashing problems.
date: "June 1 2025"
---

# Arrays & Hashing

## 1. Contains Duplicate

**Description:**  
Given an integer array `nums`, return true if any value appears at least twice in the array, and return false if every element is distinct.

**Java Solution:**
```java
public boolean containsDuplicate(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) {
        if (!set.add(num)) return true;
    }
    return false;
}
```

**Explanation:**  
We use a HashSet to track seen numbers. If we see a duplicate, we return true.

- **Time Complexity:** O(n)  
  We iterate through the array once, and each HashSet operation is O(1) on average.
- **Space Complexity:** O(n)  
  In the worst case, all elements are unique and stored in the set.

---

## 2. Valid Anagram

**Description:**  
Given two strings `s` and `t`, return true if `t` is an anagram of `s`, and false otherwise.

**Java Solution:**
```java
public boolean isAnagram(String s, String t) {
    if (s.length() != t.length()) return false;
    int[] count = new int[26];
    for (int i = 0; i < s.length(); i++) {
        count[s.charAt(i) - 'a']++;
        count[t.charAt(i) - 'a']--;
    }
    for (int c : count) if (c != 0) return false;
    return true;
}
```

**Explanation:**  
We count the frequency of each character in both strings and compare.

- **Time Complexity:** O(n)  
  We traverse both strings once.
- **Space Complexity:** O(1)  
  The count array size is constant (26 for lowercase English letters).

---

## 3. Two Sum

**Description:**  
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

**Java Solution:**
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return new int[] { map.get(complement), i };
        }
        map.put(nums[i], i);
    }
    throw new IllegalArgumentException("No two sum solution");
}
```

**Explanation:**  
We use a HashMap to store each number and its index. For each number, we check if its complement exists in the map.

- **Time Complexity:** O(n)  
  Each lookup and insertion in the map is O(1) on average.
- **Space Complexity:** O(n)  
  In the worst case, we store all n elements in the map.

---

## 4. Group Anagrams

**Description:**  
Given an array of strings, group the anagrams together.

**Java Solution:**
```java
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> map = new HashMap<>();
    for (String s : strs) {
        char[] ca = s.toCharArray();
        Arrays.sort(ca);
        String key = new String(ca);
        map.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
    }
    return new ArrayList<>(map.values());
}
```

**Explanation:**  
We sort each string to use as a key in a map. Anagrams will have the same sorted key.

- **Time Complexity:** O(n * k log k)  
  n = number of strings, k = max string length. Sorting each string is O(k log k).
- **Space Complexity:** O(nk)  
  We store all strings in the map.

---

## 5. Top K Frequent Elements

**Description:**  
Given an integer array `nums` and an integer `k`, return the k most frequent elements.

**Java Solution:**
```java
public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> count = new HashMap<>();
    for (int n : nums) count.put(n, count.getOrDefault(n, 0) + 1);
    PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> count.get(a) - count.get(b));
    for (int n : count.keySet()) {
        heap.add(n);
        if (heap.size() > k) heap.poll();
    }
    int[] res = new int[k];
    for (int i = k - 1; i >= 0; --i) res[i] = heap.poll();
    return res;
}
```

**Explanation:**  
We count frequencies, then use a min-heap to keep the top k elements.

- **Time Complexity:** O(n log k)  
  n = number of elements. Each heap operation is O(log k).
- **Space Complexity:** O(n)  
  For the frequency map and heap.

---

## 6. Product of Array Except Self

**Description:**  
Given an integer array `nums`, return an array `answer` such that `answer[i]` is the product of all the elements of `nums` except `nums[i]`.

**Java Solution:**
```java
public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] res = new int[n];
    int left = 1;
    for (int i = 0; i < n; i++) {
        res[i] = left;
        left *= nums[i];
    }
    int right = 1;
    for (int i = n - 1; i >= 0; i--) {
        res[i] *= right;
        right *= nums[i];
    }
    return res;
}
```

**Explanation:**  
We use two passes: left-to-right and right-to-left, multiplying the products before and after each index.

- **Time Complexity:** O(n)  
  Two passes through the array.
- **Space Complexity:** O(1) (excluding output array)  
  Only a few variables for products.

---

## 7. Valid Sudoku

**Description:**  
Determine if a 9x9 Sudoku board is valid.

**Java Solution:**
```java
public boolean isValidSudoku(char[][] board) {
    Set<String> seen = new HashSet<>();
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            char number = board[i][j];
            if (number != '.') {
                if (!seen.add(number + " in row " + i) ||
                    !seen.add(number + " in col " + j) ||
                    !seen.add(number + " in block " + i/3 + "-" + j/3))
                    return false;
            }
        }
    }
    return true;
}
```

**Explanation:**  
We use a set to track seen numbers in rows, columns, and blocks.

- **Time Complexity:** O(1)  
  The board size is fixed (9x9).
- **Space Complexity:** O(1)  
  The set size is bounded by the board size.

---

## 8. Encode And Decode Strings

**Description:**  
Design an algorithm to encode a list of strings to a string and decode it back.

**Java Solution:**
```java
public class Codec {
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for (String s : strs) {
            sb.append(s.length()).append('#').append(s);
        }
        return sb.toString();
    }
    public List<String> decode(String s) {
        List<String> res = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            int j = i;
            while (s.charAt(j) != '#') j++;
            int len = Integer.parseInt(s.substring(i, j));
            res.add(s.substring(j + 1, j + 1 + len));
            i = j + 1 + len;
        }
        return res;
    }
}
```

**Explanation:**  
We encode each string with its length and a separator. Decoding reads the length and extracts the string.

- **Time Complexity:** O(n)  
  n = total length of all strings.
- **Space Complexity:** O(n)  
  For the output list.

---

## 9. Longest Consecutive Sequence

**Description:**  
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

**Java Solution:**
```java
public int longestConsecutive(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) set.add(num);
    int longest = 0;
    for (int num : set) {
        if (!set.contains(num - 1)) {
            int curr = num;
            int streak = 1;
            while (set.contains(curr + 1)) {
                curr++;
                streak++;
            }
            longest = Math.max(longest, streak);
        }
    }
    return longest;
}
```

**Explanation:**  
We use a set for O(1) lookups. For each number, if it's the start of a sequence, we count the streak.

- **Time Complexity:** O(n)  
  Each number is checked at most twice.
- **Space Complexity:** O(n)  
  For the set.

--- 