---
title: Bit Manipulation
description: Java solutions with explanations, time and space complexity for Bit Manipulation problems.
date: "June 1 2025"
---

# Bit Manipulation Pattern

Bit Manipulation problems involve working with individual bits of numbers and often use:
- Bitwise operators (AND, OR, XOR, NOT)
- Bit shifting
- Bit masking
- Bit counting
- Bit manipulation tricks

## 1. Single Number (Easy)

**Problem:** Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

**Solution:**
```java
class Solution {
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;  // XOR all numbers
        }
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 2. Number of 1 Bits (Easy)

**Problem:** Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

**Solution:**
```java
public class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count += (n & 1);  // Check if last bit is 1
            n >>>= 1;  // Unsigned right shift
        }
        return count;
    }
}
```

**Time Complexity:** O(1) - as we process at most 32 bits
**Space Complexity:** O(1)

## 3. Counting Bits (Easy)

**Problem:** Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

**Solution:**
```java
class Solution {
    public int[] countBits(int n) {
        int[] result = new int[n + 1];
        
        for (int i = 1; i <= n; i++) {
            // i & (i-1) removes the least significant 1
            result[i] = result[i & (i-1)] + 1;
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 4. Reverse Bits (Easy)

**Problem:** Reverse bits of a given 32 bits unsigned integer.

**Solution:**
```java
public class Solution {
    public int reverseBits(int n) {
        int result = 0;
        
        for (int i = 0; i < 32; i++) {
            result <<= 1;  // Shift left
            result |= (n & 1);  // Add last bit of n
            n >>>= 1;  // Shift n right
        }
        
        return result;
    }
}
```

**Time Complexity:** O(1) - as we process exactly 32 bits
**Space Complexity:** O(1)

## 5. Missing Number (Easy)

**Problem:** Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

**Solution:**
```java
class Solution {
    public int missingNumber(int[] nums) {
        int result = nums.length;
        
        for (int i = 0; i < nums.length; i++) {
            result ^= i ^ nums[i];
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 6. Sum of Two Integers (Medium)

**Problem:** Given two integers a and b, return the sum of the two integers without using the operators + and -.

**Solution:**
```java
class Solution {
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;  // Calculate carry
            a = a ^ b;  // Sum without carry
            b = carry;  // Update b to carry
        }
        return a;
    }
}
```

**Time Complexity:** O(1) - as we process at most 32 bits
**Space Complexity:** O(1)

## 7. Reverse Integer (Medium)

**Problem:** Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-2³¹, 2³¹ - 1], then return 0.

**Solution:**
```java
class Solution {
    public int reverse(int x) {
        int result = 0;
        
        while (x != 0) {
            int digit = x % 10;
            
            // Check for overflow
            if (result > Integer.MAX_VALUE/10 || 
                (result == Integer.MAX_VALUE/10 && digit > 7)) return 0;
            if (result < Integer.MIN_VALUE/10 || 
                (result == Integer.MIN_VALUE/10 && digit < -8)) return 0;
            
            result = result * 10 + digit;
            x /= 10;
        }
        
        return result;
    }
}
```

**Time Complexity:** O(log x)
**Space Complexity:** O(1)

## Key Takeaways

1. Bit Manipulation is perfect for:
   - Working with individual bits
   - Optimizing space usage
   - Fast arithmetic operations
   - Bit-level operations
   - Memory-efficient solutions

2. Common patterns:
   - XOR for finding unique numbers
   - Bit shifting for multiplication/division
   - Bit masking for specific bit operations
   - Bit counting techniques
   - Overflow handling

3. Tips:
   - Remember bitwise operator precedence
   - Consider signed vs unsigned operations
   - Watch out for overflow
   - Use appropriate bit masks
   - Think about bit-level optimizations 