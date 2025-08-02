---
title: Stack
description: Java solutions with explanations, time and space complexity for Stack problems from Blind 75.
date: "June 1 2025"
order: 4
---

# Stack

This section covers problems that can be efficiently solved using stack data structures. Stack problems often involve matching pairs, tracking history, or maintaining order.

## 1. Valid Parentheses (Easy)

**Problem:** Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

**Example:**
```
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false
```

**Solution:**
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
}
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Alternative Approach (Using HashMap):**
```java
class Solution {
    public boolean isValid(String s) {
        Map<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put('}', '{');
        map.put(']', '[');
        
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (!map.containsKey(c)) {
                stack.push(c);
            } else {
                if (stack.isEmpty() || stack.pop() != map.get(c)) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
}
```

This approach uses a HashMap to map closing brackets to their corresponding opening brackets, making the code more concise and maintainable.

## Key Takeaways

1. **Stack Operations**: Use push/pop operations to maintain bracket matching
2. **Order Matters**: Stacks naturally maintain the LIFO order needed for bracket matching
3. **Edge Cases**: Always check for empty stack before popping
4. **HashMap Mapping**: Use hash maps to simplify bracket matching logic
5. **Final Check**: Ensure stack is empty after processing all characters 