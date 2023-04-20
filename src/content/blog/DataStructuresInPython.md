---
title: "Data Structures in Python"
description: "An overview of commonly used data structures in Python..."
pubDate: "Apr 20 2023"
heroImage: "/post_img.webp"
---
Data structures are an essential part of programming that allow developers to organize and manage data efficiently. In Python, there are several built-in data structures that are commonly used in programming. In this blog post, we will explore some of the most commonly used data structures in Python, and provide examples to illustrate their usage.

**Lists**  
Lists are one of the most commonly used data structures in Python. A list is an ordered collection of elements, which can be of any data type. Lists are mutable, meaning that their elements can be modified after they are created.  
To create a list, you can use the square brackets notation, like this:
```python
my_list = [1, 2, 3, "four", 5.6]
```
To access elements of a list, you can use indexing. The first element of a list is at index 0, the second element is at index 1, and so on. For example, to access the first element of my_list, you can use the following code:

```python
print(my_list[0])  # Output: 1
```
To modify an element of a list, you can use indexing as well. For example, to change the second element of my_list to "two", you can use the following code:

```python
my_list[1] = "two"
print(my_list)  # Output: [1, 'two', 3, 'four', 5.6]
```
**Tuples**  
Tuples are similar to lists, but they are immutable. Once a tuple is created, its elements cannot be modified. Tuples are often used to represent a collection of related values that should not be changed.

To create a tuple, you can use parentheses instead of square brackets, like this:

```python
my_tuple = (1, 2, 3, "four", 5.6)
```
To access elements of a tuple, you can use indexing in the same way as with lists. For example, to access the third element of my_tuple, you can use the following code:

```python
print(my_tuple[2])  # Output: 3
```
Because tuples are immutable, you cannot modify their elements. For example, the following code will raise a TypeError:

```python
my_tuple[1] = "two"  # Raises TypeError: 'tuple' object does not support item assignment
```
**Sets**  
A set is an unordered collection of unique elements. Sets are commonly used to perform mathematical operations like union, intersection, and difference.

To create a set, you can use curly braces or the set() function. For example:

```python
my_set = {1, 2, 3, 4}
my_set = set([1, 2, 3, 4])
```
To add an element to a set, you can use the add() method. For example:

```python
my_set.add(5)
```
To remove an element from a set, you can use the remove() method. For example:

```python
my_set.remove(3)
```
**Dictionaries**  
A dictionary is an unordered collection of key-value pairs. Dictionaries are commonly used to represent data that can be accessed using a unique key.

To create a dictionary, you can use curly braces and colons to separate keys and values, like this:

```python
my_dict = {"name": "Alice", "age": 25, "city": "New York"}
```
To access a value in a dictionary, you can use the key as an index. For example:

```python
print(my_dict["age"])  # Output: 25
```
To add a new key-value pair to a dictionary, you can simply assign a value to a new key, like this:

```python
my_dict["gender"] = "female"
```
To remove a key-value pair from a dictionary, you can use the del keyword, like this:

```python
del my_dict["city"]
```
**Stacks**  
A stack is a Last-In-First-Out (LIFO) data structure, which means that the last element added to the stack will be the first one removed. Stacks are commonly used to perform operations like undo/redo or to evaluate expressions.

In Python, you can use a list as a stack by using the append() method to add elements to the top of the stack and the pop() method to remove elements from the top of the stack. For example:

```python
my_stack = []
my_stack.append(1)
my_stack.append(2)
my_stack.append(3)
print(my_stack.pop())  # Output: 3
print(my_stack.pop())  # Output: 2
```
**Queues**  
A queue is a First-In-First-Out (FIFO) data structure, which means that the first element added to the queue will be the first one removed. Queues are commonly used to manage tasks that need to be executed in the order they were received.

In Python, you can use the deque class from the collections module to implement a queue. The deque class provides the append() method to add elements to the end of the queue and the popleft() method to remove elements from the beginning of the queue. For example:

```python
from collections import deque

my_queue = deque()
my_queue.append(1)
my_queue.append(2)
my_queue.append(3)
print(my_queue.popleft())  # Output: 1
print(my_queue.popleft())  # Output: 2
```
In this blog post, we have explored some of the most commonly used data structures in Python. Each data structure has its own characteristics and usage patterns, and choosing the right one for a particular task can greatly improve the efficiency and readability of your code. By understanding these data structures and their corresponding methods, you can write more effective and efficient Python code.
