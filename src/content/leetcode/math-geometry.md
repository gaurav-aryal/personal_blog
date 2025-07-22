---
title: Math & Geometry
description: Java solutions with explanations, time and space complexity for Math & Geometry problems.
date: "June 1 2025"
---

# Math & Geometry Pattern

Math and Geometry problems often involve:
- Matrix operations
- Number theory
- Geometric transformations
- Coordinate systems
- Mathematical formulas
- Pattern recognition

## 1. Rotate Image (Medium)

**Problem:** You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

**Solution:**
```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        
        // Transpose the matrix
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        // Reverse each row
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n/2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n-1-j];
                matrix[i][n-1-j] = temp;
            }
        }
    }
}
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)

## 2. Spiral Matrix (Medium)

**Problem:** Given an m x n matrix, return all elements of the matrix in spiral order.

**Solution:**
```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return result;
        
        int top = 0, bottom = matrix.length - 1;
        int left = 0, right = matrix[0].length - 1;
        
        while (top <= bottom && left <= right) {
            // Traverse right
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            top++;
            
            // Traverse down
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            right--;
            
            // Traverse left
            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
                bottom--;
            }
            
            // Traverse up
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
                left++;
            }
        }
        
        return result;
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(1)

## 3. Set Matrix Zeroes (Medium)

**Problem:** Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

**Solution:**
```java
class Solution {
    public void setZeroes(int[][] matrix) {
        boolean firstRow = false;
        boolean firstCol = false;
        
        // Check if first row has any zero
        for (int j = 0; j < matrix[0].length; j++) {
            if (matrix[0][j] == 0) {
                firstRow = true;
                break;
            }
        }
        
        // Check if first column has any zero
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                firstCol = true;
                break;
            }
        }
        
        // Use first row and column as markers
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        
        // Set zeros based on markers
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // Set first row to zero if needed
        if (firstRow) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[0][j] = 0;
            }
        }
        
        // Set first column to zero if needed
        if (firstCol) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(1)

## 4. Happy Number (Easy)

**Problem:** Write an algorithm to determine if a number n is happy. A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1.

**Solution:**
```java
class Solution {
    public boolean isHappy(int n) {
        Set<Integer> seen = new HashSet<>();
        
        while (n != 1 && !seen.contains(n)) {
            seen.add(n);
            n = getNext(n);
        }
        
        return n == 1;
    }
    
    private int getNext(int n) {
        int sum = 0;
        while (n > 0) {
            int digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }
        return sum;
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(log n)

## 5. Plus One (Easy)

**Problem:** You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order.

**Solution:**
```java
class Solution {
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        
        for (int i = n - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            }
            digits[i] = 0;
        }
        
        // If we're here, we need a new array with a leading 1
        int[] newDigits = new int[n + 1];
        newDigits[0] = 1;
        return newDigits;
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

## 6. Pow(x, n) (Medium)

**Problem:** Implement pow(x, n), which calculates x raised to the power n.

**Solution:**
```java
class Solution {
    public double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n == Integer.MIN_VALUE) {
            x = x * x;
            n = n/2;
        }
        if (n < 0) {
            x = 1/x;
            n = -n;
        }
        
        return (n % 2 == 0) ? myPow(x * x, n/2) : x * myPow(x * x, n/2);
    }
}
```

**Time Complexity:** O(log n)
**Space Complexity:** O(log n)

## 7. Multiply Strings (Medium)

**Problem:** Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

**Solution:**
```java
class Solution {
    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] pos = new int[m + n];
        
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i + j, p2 = i + j + 1;
                int sum = mul + pos[p2];
                
                pos[p1] += sum / 10;
                pos[p2] = sum % 10;
            }
        }
        
        StringBuilder sb = new StringBuilder();
        for (int p : pos) {
            if (!(sb.length() == 0 && p == 0)) {
                sb.append(p);
            }
        }
        
        return sb.length() == 0 ? "0" : sb.toString();
    }
}
```

**Time Complexity:** O(m * n)
**Space Complexity:** O(m + n)

## 8. Detect Squares (Medium)

**Problem:** You are given a stream of points on the X-Y plane. Design an algorithm that:
- Adds new points from the stream into a data structure
- Counts the number of ways to pick three points that form an axis-aligned square

**Solution:**
```java
class DetectSquares {
    private Map<Integer, Map<Integer, Integer>> points;
    
    public DetectSquares() {
        points = new HashMap<>();
    }
    
    public void add(int[] point) {
        int x = point[0], y = point[1];
        points.putIfAbsent(x, new HashMap<>());
        Map<Integer, Integer> yMap = points.get(x);
        yMap.put(y, yMap.getOrDefault(y, 0) + 1);
    }
    
    public int count(int[] point) {
        int x = point[0], y = point[1];
        int count = 0;
        
        if (!points.containsKey(x)) return 0;
        
        Map<Integer, Integer> yMap = points.get(x);
        for (int y1 : yMap.keySet()) {
            if (y1 == y) continue;
            
            int len = Math.abs(y1 - y);
            
            // Check for squares in both directions
            count += yMap.get(y1) * 
                    points.getOrDefault(x + len, new HashMap<>()).getOrDefault(y, 0) *
                    points.getOrDefault(x + len, new HashMap<>()).getOrDefault(y1, 0);
                    
            count += yMap.get(y1) * 
                    points.getOrDefault(x - len, new HashMap<>()).getOrDefault(y, 0) *
                    points.getOrDefault(x - len, new HashMap<>()).getOrDefault(y1, 0);
        }
        
        return count;
    }
}
```

**Time Complexity:** 
- Add: O(1)
- Count: O(n)
**Space Complexity:** O(n)

## Key Takeaways

1. Math & Geometry problems often involve:
   - Matrix operations and transformations
   - Number theory concepts
   - Geometric calculations
   - Pattern recognition
   - Mathematical formulas

2. Common patterns:
   - Matrix traversal
   - Coordinate system manipulation
   - Number manipulation
   - Geometric transformations
   - Mathematical optimization

3. Tips:
   - Consider edge cases
   - Use appropriate data structures
   - Optimize for space when possible
   - Think about mathematical properties
   - Consider geometric relationships 