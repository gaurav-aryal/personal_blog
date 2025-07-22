---
title: Stack
description: Java solutions with explanations, time and space complexity for Stack problems.
date: "June 1 2025"
---

# Stack Pattern

The Stack pattern is a fundamental data structure that follows Last-In-First-Out (LIFO) principle. It's particularly useful for problems involving:
- Parentheses matching
- Expression evaluation
- Next greater/smaller element
- Histogram problems
- Browser history

## 1. Valid Parentheses (Easy)

**Problem:** Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

**Solution:**
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();

        //map containing character
        HashMap<Character, Character> map = new HashMap<>();
        map.put('}','{');
        map.put(']','[');
        map.put(')','(');

        for(char c: s.toCharArray()){
            if(map.containsKey(c)){
                //closed parenthesis
                //check if stack contains open parenthesis
                if(stack.isEmpty() || stack.pop() != map.get(c)){
                    return false;
                }
            } else {
                //push all open parenthesis into the stack
                //so that when closed is found, we can check stack to make sure 
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }
}
```

**Time Complexity:** O(n) where n is the length of the string
**Space Complexity:** O(n) for the stack

## 2. Min Stack (Medium)

**Problem:** Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

**Solution:**
```java
class MinStack {

    private Stack<Integer> stack;
    private Stack<Integer> minStack;

    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int val) {
        stack.push(val);
        if(minStack.isEmpty() || val <= minStack.peek()){
            minStack.push(val);
        }
    }
    
    public void pop() {
        if (!stack.isEmpty()) {
            int popped = stack.pop();
            if (popped == minStack.peek()) {
                minStack.pop();
            }
        }
    }
    
    public int top() {
        if(!stack.isEmpty()){
            return stack.peek();
        }
        throw new IllegalStateException("Stack is empty");
    }
    
    public int getMin() {
        if (!minStack.isEmpty()) {
            return minStack.peek();
        }
        throw new IllegalStateException("Stack is empty");
    }
}
```

**Time Complexity:** O(1) for all operations
**Space Complexity:** O(n) where n is the number of elements

## 3. Evaluate Reverse Polish Notation (Medium)

**Problem:** Evaluate the value of an arithmetic expression in Reverse Polish Notation.

**Solution:**
```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        
        for (String token : tokens) {
            if (token.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (token.equals("-")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a - b);
            } else if (token.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (token.equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a / b);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        
        return stack.pop();
    }
}
```

**Time Complexity:** O(n) where n is the number of tokens
**Space Complexity:** O(n) for the stack

## 4. Generate Parentheses (Medium)

**Problem:** Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Solution:**
```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        backtrack(result, "", 0, 0, n);
        return result;
    }
    
    private void backtrack(List<String> result, String current, int open, int close, int max) {
        if (current.length() == max * 2) {
            result.add(current);
            return;
        }
        
        if (open < max) {
            backtrack(result, current + "(", open + 1, close, max);
        }
        if (close < open) {
            backtrack(result, current + ")", open, close + 1, max);
        }
    }
}
```

**Time Complexity:** O(4^n/√n) - Catalan number
**Space Complexity:** O(n) for recursion stack

## 5. Daily Temperatures (Medium)

**Problem:** Given an array of integers temperatures representing daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.

**Solution:**
```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int prev = stack.pop();
                result[prev] = i - prev;
            }
            stack.push(i);
        }
        
        return result;
    }
}
```

**Time Complexity:** O(n) where n is the length of temperatures array
**Space Complexity:** O(n) for the stack

## 6. Car Fleet (Medium)

**Problem:** There are n cars going to the same destination along a one-lane road. The destination is target miles away. You are given two integer arrays position and speed, where position[i] is the position of the ith car and speed[i] is the speed of the ith car. A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same speed. The distance between these two cars is ignored. Return the number of car fleets that will arrive at the destination.

**Solution:**
```java
class Solution {
    public int carFleet(int target, int[] position, int[] speed) {
        int n = position.length;
        double[][] cars = new double[n][2];
        
        for (int i = 0; i < n; i++) {
            cars[i] = new double[]{position[i], (double)(target - position[i]) / speed[i]};
        }
        
        Arrays.sort(cars, (a, b) -> Double.compare(b[0], a[0]));
        
        Stack<Double> stack = new Stack<>();
        for (double[] car : cars) {
            if (stack.isEmpty() || car[1] > stack.peek()) {
                stack.push(car[1]);
            }
        }
        
        return stack.size();
    }
}
```

**Time Complexity:** O(n log n) for sorting
**Space Complexity:** O(n) for the stack

## 7. Largest Rectangle in Histogram (Hard)

**Problem:** Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Solution:**
```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        
        for (int i = 0; i <= n; i++) {
            int h = (i == n) ? 0 : heights[i];
            
            while (!stack.isEmpty() && h < heights[stack.peek()]) {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            
            stack.push(i);
        }
        
        return maxArea;
    }
}
```

**Time Complexity:** O(n) where n is the length of heights array
**Space Complexity:** O(n) for the stack

## Key Takeaways

1. Stack is perfect for problems involving:
   - Matching pairs (parentheses, brackets)
   - Next greater/smaller element
   - Expression evaluation
   - Histogram problems

2. Common patterns:
   - Use stack to keep track of indices or values
   - Process elements in reverse order
   - Maintain monotonic stack (increasing or decreasing)

3. Tips:
   - Consider edge cases (empty stack)
   - Think about what to store in the stack (indices vs values)
   - Use stack for problems requiring LIFO order 