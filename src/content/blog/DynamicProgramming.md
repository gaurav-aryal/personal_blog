---
title: "Dynamic Programming"
description: "The concept of dynamic programming and Python code examples..."
pubDate: "Apr 17 2023"
heroImage: "/post_img.webp"
---
Dynamic programming is a popular technique used in computer science to optimize problems with overlapping subproblems. In this blog post, we will explore dynamic programming in detail and provide Python code examples to demonstrate its usage.

What is Dynamic Programming?  
Dynamic programming is a technique used to solve optimization problems by breaking them down into smaller subproblems and reusing the solutions of these subproblems to solve the larger problem. It is used when the solution to a problem can be obtained by combining the solutions of smaller subproblems.  
Dynamic programming is similar to divide and conquer, in that both techniques break down a problem into smaller subproblems. However, unlike divide and conquer, dynamic programming reuses the solutions of smaller subproblems to solve larger subproblems.  
Dynamic programming is useful for problems with overlapping subproblems, where the same subproblems are repeatedly encountered. By storing the solutions of subproblems in a table, dynamic programming can reduce the time complexity of the algorithm.

Steps of Dynamic Programming  
The following steps are involved in solving a problem using dynamic programming:  
1. Characterize the structure of an optimal solution.
2. Define the value of the optimal solution recursively.
3. Compute the value of the optimal solution in a bottom-up fashion.
4. Construct an optimal solution from the computed information.
5. Dynamic programming can be implemented using either a top-down (memoization) or bottom-up (tabulation) approach.

**Top-Down Approach (Memoization)**  
The top-down approach, also known as memoization, involves solving the problem recursively by storing the solutions of subproblems in a memo table. If a subproblem is encountered again, its solution is retrieved from the memo table instead of recalculating it. Memoization can greatly reduce the time complexity of the algorithm by avoiding repeated calculations.  
Here is an example of the Fibonacci sequence using memoization:  
```python
memo = {}

def fibonacci(n):
    if n in memo:
        return memo[n]
    elif n <= 2:
        return 1
    else:
        memo[n] = fibonacci(n-1) + fibonacci(n-2)
        return memo[n]

print(fibonacci(10))
```

**Bottom-Up Approach (Tabulation)**  
The bottom-up approach, also known as tabulation, involves solving the problem iteratively by storing the solutions of subproblems in a table. The table is filled in a bottom-up fashion, starting with the smallest subproblems and working up to the largest subproblems. Tabulation can be more efficient than memoization in terms of space complexity, as it does not require the additional overhead of function calls.  
Here is an example of the Fibonacci sequence using tabulation:  
```python
def fibonacci(n):
    if n <= 2:
        return 1
    else:
        fib = [0] * (n+1)
        fib[1] = 1
        fib[2] = 1
        for i in range(3, n+1):
            fib[i] = fib[i-1] + fib[i-2]
        return fib[n]

print(fibonacci(10))
```

**Dynamic Programming Examples**  
**Longest Common Subsequence**  
The longest common subsequence (LCS) problem involves finding the longest subsequence that is present in two given sequences. A subsequence is a sequence that can be derived from another sequence by deleting some elements without changing the order of the remaining elements.  
Here is the Python code for the LCS problem using dynamic programming:  
```python
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    lcs_table = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                lcs_table[i][j] = lcs_table[i-1][j-1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])
    
    lcs = ""
    i, j = m, n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs = X[i-1] + lcs
            i -= 1
            j -= 1
        elif lcs_table[i-1][j] > lcs_table[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return lcs
```
This code uses a two-dimensional list to store the length of the longest common subsequence between the prefixes of X and Y, and then backtracks through this list to find the actual subsequence.

**Knapsack Problem**  
The knapsack problem is a well-known optimization problem that involves finding the maximum value that can be obtained by selecting a subset of items with a given weight limit.  
**0/1 Knapsack**  
In the 0/1 knapsack problem, we are given a set of items, each with a weight and a value, and a knapsack that can hold a maximum weight limit. We want to select a subset of items such that the total weight does not exceed the weight limit of the knapsack and the total value is maximum. The 0/1 in the name of the problem refers to the fact that each item can either be taken (1) or not taken (0).  
Here is the Python code for solving the 0/1 knapsack problem using dynamic programming:  
```python
def knapsack_01(weights, values, max_weight):
    n = len(weights)
    dp = [[0 for j in range(max_weight + 1)] for i in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, max_weight + 1):
            if weights[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]])
    
    return dp[n][max_weight]
    ```

The knapsack_01 function takes three arguments: weights and values, which are lists of weights and values of the items, and max_weight, which is the maximum weight limit of the knapsack. The function creates a two-dimensional list dp of size (n + 1) × (max_weight + 1), where n is the number of items. The value of dp[i][j] represents the maximum value that can be obtained by selecting a subset of the first i items with a total weight of at most j. The function initializes the first row and the first column of dp to 0, since selecting 0 items or having a maximum weight of 0 will always result in a total value of 0. The function then fills in the remaining entries of dp using the recurrence relation:
```python
dp[i][j] = max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]])
```
This recurrence relation states that the maximum value that can be obtained by selecting a subset of the first i items with a total weight of at most j is either the maximum value that can be obtained by selecting a subset of the first i - 1 items with a total weight of at most j (i.e., not selecting the ith item), or the value of the ith item plus the maximum value that can be obtained by selecting a subset of the first i - 1 items with a total weight of at most j - weights[i - 1] (i.e., selecting the ith item).  
The time complexity of the 0/1 knapsack problem using dynamic programming is O(nW), where n is the number of items and W is the maximum weight limit of the knapsack. The space complexity is also O(nW), since we need to store the two-dimensional list dp.

**Unbounded Knapsack**  
In the unbounded knapsack problem, we are given a knapsack with a maximum weight limit and a set of items, each with a weight and a value. Unlike the 0/1 knapsack problem, we can take multiple instances of an item in the unbounded knapsack problem.
The goal is to find the maximum possible value that can be obtained by filling the knapsack with the items.  
Here is the Python code for the unbounded knapsack problem using dynamic programming:  
```python
def unbounded_knapsack(W, wt, val):
    n = len(wt)
    dp = [0] * (W+1)
    
    for w in range(1, W+1):
        for i in range(n):
            if wt[i] <= w:
                dp[w] = max(dp[w], dp[w-wt[i]] + val[i])
    
    return dp[W]
```
In this code, W is the maximum weight limit of the knapsack, wt is the list of weights of the items, and val is the list of values of the items.  
The dp list is initialized to all zeroes, and for each weight w from 1 to W, we iterate over all items and update dp[w] if the current item's weight is less than or equal to w. The value of dp[w-wt[i]] + val[i] represents the maximum value that can be obtained if we include the current item, and we take the maximum of this value and dp[w].  
The time complexity of the unbounded knapsack problem using dynamic programming is O(nW), where n is the number of items and W is the maximum weight limit of the knapsack. The space complexity is O(W), since we only need to store a one-dimensional list dp.