---
title: Sliding Window
description: Java solutions with explanations, time and space complexity for Sliding Window problems.
date: "June 1 2025"
---

# Sliding Window

## 1. Best Time to Buy And Sell Stock

**Description:**  
You are given an array `prices` where `prices[i]` is the price of a given stock on the `i`th day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

**Java Solution:**
```java
class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else if (prices[i] - minPrice > maxProfit) {
                maxProfit = prices[i] - minPrice;
            }
        }
        return maxProfit;
    }
}
```

**Explanation:**  
We iterate through the prices, keeping track of the minimum price encountered so far. At each step, we calculate the potential profit if we were to sell on the current day (current price - `minPrice`) and update `maxProfit` if this profit is greater.

- **Time Complexity:** O(n)  
  We iterate through the array once.
- **Space Complexity:** O(1)  
  Only a few variables are used.

---

## 2. Longest Substring Without Repeating Characters

**Description:**  
Given a string `s`, find the length of the longest substring without repeating characters.

**Java Solution:**
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> charMap = new HashMap<>();
        int maxLength = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            char currentChar = s.charAt(right);
            if (charMap.containsKey(currentChar)) {
                left = Math.max(left, charMap.get(currentChar) + 1);
            }
            charMap.put(currentChar, right);
            maxLength = Math.max(maxLength, right - left + 1);
        }
        return maxLength;
    }
}
```

**Explanation:**  
We use a sliding window approach with a HashMap to store the last seen index of each character. If a duplicate is found within the current window, we move the `left` pointer to ensure the window contains no repeating characters.

- **Time Complexity:** O(n)  
  Both `left` and `right` pointers traverse the string at most once. HashMap operations (put, containsKey, get) take O(1) on average.
- **Space Complexity:** O(min(n, m))  
  Where `n` is the string length and `m` is the size of the character set (e.g., 26 for English lowercase letters, 128 for ASCII, 256 for extended ASCII, or Unicode character set size). In the worst case, all characters in the current window are unique and stored in the HashMap.

---

## 3. Longest Repeating Character Replacement

**Description:**  
You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character any number of times to make a longest repeating character substring. Return the length of the longest such substring.

**Java Solution:**
```java
class Solution {
    public int characterReplacement(String s, int k) {
        int[] charCounts = new int[26];
        int maxLength = 0;
        int maxFrequency = 0; // Frequency of the most frequent character in the current window
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            charCounts[s.charAt(right) - 'A']++;
            maxFrequency = Math.max(maxFrequency, charCounts[s.charAt(right) - 'A']);

            // Window size - count of most frequent char > k, then shrink window
            while ((right - left + 1) - maxFrequency > k) {
                charCounts[s.charAt(left) - 'A']--;
                left++;
            }
            maxLength = Math.max(maxLength, right - left + 1);
        }
        return maxLength;
    }
}
```

**Explanation:**  
This uses a sliding window. We expand the window from `left` to `right`. We keep track of the frequency of each character within the window. If the number of characters to change (window size - `maxFrequency`) exceeds `k`, we shrink the window from the `left`.

- **Time Complexity:** O(n)  
  Both pointers traverse the string once. Character frequency updates and max frequency checks are constant time.
- **Space Complexity:** O(1)  
  The `charCounts` array size is constant (26 for uppercase English letters).

---

## 4. Permutation In String

**Description:**  
Given two strings `s1` and `s2`, return `true` if `s2` contains a permutation of `s1`, or `false` otherwise.

**Java Solution:**
```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) return false;

        int[] s1Count = new int[26];
        int[] s2Count = new int[26];

        // Initialize counts for the first window of s2
        for (int i = 0; i < s1.length(); i++) {
            s1Count[s1.charAt(i) - 'a']++;
            s2Count[s2.charAt(i) - 'a']++;
        }

        int matches = 0;
        for (int i = 0; i < 26; i++) {
            if (s1Count[i] == s2Count[i]) {
                matches++;
            }
        }

        int left = 0;
        for (int right = s1.length(); right < s2.length(); right++) {
            if (matches == 26) return true; // Found a permutation

            // Add new character to window
            int charToAdd = s2.charAt(right) - 'a';
            s2Count[charToAdd]++;
            if (s1Count[charToAdd] == s2Count[charToAdd]) {
                matches++;
            } else if (s1Count[charToAdd] + 1 == s2Count[charToAdd]) {
                matches--;
            }

            // Remove old character from window
            int charToRemove = s2.charAt(left) - 'a';
            s2Count[charToRemove]--;
            if (s1Count[charToRemove] == s2Count[charToRemove]) {
                matches++;
            } else if (s1Count[charToRemove] - 1 == s2Count[charToRemove]) {
                matches--;
            }
            left++;
        }
        return matches == 26; // Check the last window
    }
}
```

**Explanation:**  
We use a sliding window of size `s1.length()`. We maintain character counts for `s1` and the current window of `s2`. We also keep a `matches` counter to track how many characters have matching frequencies. When `matches` reaches 26 (for all lowercase English letters), a permutation is found.

- **Time Complexity:** O(L1 + L2)  
  Where L1 is the length of `s1` and L2 is the length of `s2`. We iterate through `s1` once to get initial counts, and then iterate through `s2` once with the sliding window.
- **Space Complexity:** O(1)  
  The count arrays have a fixed size of 26.

---

## 5. Minimum Window Substring

**Description:**  
Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return the empty string `""`.

**Java Solution:**
```java
class Solution {
    public String minWindow(String s, String t) {
        if(s == null || s.length() == 0 || s.length() < t.length()) return "";

        Map<Character, Integer> tfreq = new HashMap<>();
        for (char c : t.toCharArray()) {
            tfreq.put(c, tfreq.getOrDefault(c, 0) + 1);
        }

        int left = 0;
        int minLen = Integer.MAX_VALUE;
        int minStart = 0;
        int matchCount = 0;

        Map<Character, Integer> window = new HashMap<>();

        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);
            window.put(c, window.getOrDefault(c, 0) + 1);

            //check if the tFreq map contains the char and the count matches
            if(tfreq.containsKey(c) && tfreq.get(c).intValue()==window.get(c).intValue()){
                matchCount++;
            }

            while (matchCount==tfreq.size()){
                //update the window's left and window's diff
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }

                //shrink the window to find potentially smaller window
                char leftChar = s.charAt(left);
                window.put(leftChar, window.get(leftChar) - 1);

                if (tfreq.containsKey(leftChar) && window.get(leftChar).intValue() < tfreq.get(leftChar).intValue()) {
                    matchCount--;
                }
                left++;
            }
        }
        return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
    }
}
```

**Explanation:**  
This is a classic sliding window problem. We use two pointers (`left` and `right`) and two HashMaps: one for required character counts in `t` and another for current character counts in the window. We expand the window until all characters in `t` are covered (`formed == required`). Then, we shrink the window from the left to find the minimum valid window.

- **Time Complexity:** O(S + T)  
  Where S is the length of `s` and T is the length of `t`. The `right` pointer iterates through `s` once. The `left` pointer iterates through `s` once. HashMap operations are O(1) on average.
- **Space Complexity:** O(1)  
  The HashMaps store at most the number of unique characters in the input strings. Since the character set is typically constant (e.g., 256 for ASCII), the space complexity is O(1).

---

## 6. Sliding Window Maximum

**Description:**  
You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

**Java Solution:**
```java
import java.util.Deque;
import java.util.LinkedList;

class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || k <= 0) {
            return new int[0];
        }

        int n = nums.length;
        int[] result = new int[n - k + 1];
        int resultIndex = 0;

        // Stores indices
        Deque<Integer> deque = new LinkedList<>();

        for (int i = 0; i < n; i++) {
            // Remove indices out of window from front
            if (!deque.isEmpty() && deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }

            // Remove smaller elements from back
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }

            deque.offerLast(i);

            // Add to result when window is formed
            if (i >= k - 1) {
                result[resultIndex++] = nums[deque.peekFirst()];
            }
        }
        return result;
    }
}
```

**Explanation:**  
We use a `Deque` (double-ended queue) to maintain indices of elements in decreasing order within the current window. This ensures that the front of the deque always holds the index of the maximum element in the current window. As the window slides, we remove old indices and add new ones, maintaining the decreasing order.

- **Time Complexity:** O(n)  
  Each element is added to and removed from the deque at most once. Therefore, each operation in the loop takes amortized O(1) time.
- **Space Complexity:** O(k)  
  The deque stores at most `k` elements (indices) at any given time.

--- 