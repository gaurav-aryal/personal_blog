---
title: "Functional Programming"
description: "A programming paradigm that emphasizes the use of pure functions to solve problems..."
pubDate: "Mar 24 2023"
heroImage: "/post_img.webp"
---

Functional programming is a programming paradigm that emphasizes the use of pure functions to solve problems. In functional programming, functions are treated as first-class citizens and are used to compose larger solutions. In this blog post, we will explore functional programming concepts and how they can be applied in Python.

**Pure functions**  
A pure function is a function that always returns the same output for the same input and does not have any side effects. A side effect is any modification of state that is visible outside of the function, such as modifying a global variable or printing to the console. Pure functions are easy to reason about, test and compose. Here's an example of a pure function that computes the area of a circle:

```python
import math

def circle_area(radius):
    return math.pi * radius ** 2
```

**Higher-order functions**  
A higher-order function is a function that takes one or more functions as arguments or returns a function as its result. Higher-order functions allow us to abstract over patterns of computation and express complex operations in a concise and modular way. Here's an example of a higher-order function that applies a function to each element of a list:

```python
def map(func, lst):
    return [func(x) for x in lst]
```

We can use the map function to apply a function to each element of a list:

```python
def square(x):
    return x ** 2
lst = [1, 2, 3, 4, 5]
squared_lst = map(square, lst)
print(squared_lst)  # [1, 4, 9, 16, 25]
```

**Lambda functions**  
A lambda function is a small anonymous function that can be defined inline. Lambda functions are useful when we need to pass a simple function as an argument to a higher-order function without defining a separate function. Here's an example of a lambda function that adds two numbers:

```python
add = lambda x, y: x + y
```

We can use the lambda function with the map function to add two numbers:

```python
lst1 = [1, 2, 3]
lst2 = [4, 5, 6]
added_lst = map(lambda x, y: x + y, lst1, lst2)
print(added_lst)  # [5, 7, 9]
```

**Filtering**  
Filtering is a common operation in functional programming that selects elements from a collection based on a predicate. We can use the filter function to select elements from a list that satisfy a condition:

```python
def is_even(x):
    return x % 2 == 0

lst = [1, 2, 3, 4, 5]
even_lst = filter(is_even, lst)
print(even_lst)  # [2, 4]
```

**Reducing**  
Reducing is a common operation in functional programming that combines the elements of a collection into a single value. We can use the reduce function to apply a binary function to the elements of a collection in a cumulative way:

```python
from functools import reduce

def add(x, y):
    return x + y

lst = [1, 2, 3, 4, 5]
sum_lst = reduce(add, lst)
print(sum_lst)  # 15
```
