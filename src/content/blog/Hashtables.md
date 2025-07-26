---
title: "Understanding Hashtables"
description: "the concept of hashtables, their practical implementation in Python, and how they enable efficient key-value storage and retrieval..."
pubDate: "May 31 2023"
heroImage: "/post_img.webp"
---
**Introduction:**  
Hashtables are fundamental data structures in computer science that provide efficient key-value storage and retrieval. In this blog post, we will explore the concept of hashtables, their underlying principles, and how to leverage them in Python to solve real-world problems. We will also provide a practical implementation of a hashtable using Python code.

**Understanding Hashtables:**  
A hashtable is a data structure that enables fast retrieval of values based on a unique key. It achieves this by using a hashing function to map the keys to specific indices in an array, known as buckets or slots. Each slot holds a reference to the corresponding value.

The key advantage of hashtables is their ability to perform key-value lookups in constant time on average, regardless of the size of the dataset. This makes hashtables suitable for scenarios where fast access to values based on keys is essential.

**Implementing a Hashtable in Python:**  
To demonstrate the implementation of a hashtable in Python, we will create a basic version that supports key insertion, retrieval, and deletion operations. Here's an example of a Hashtable class:
```python
class Hashtable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        slot = self.table[index]
        for i, (k, v) in enumerate(slot):
            if k == key:
                slot[i] = (key, value)  # Update existing key-value pair
                return
        slot.append((key, value))  # Add new key-value pair

    def get(self, key):
        index = self._hash(key)
        slot = self.table[index]
        for k, v in slot:
            if k == key:
                return v
        raise KeyError("Key not found")

    def delete(self, key):
        index = self._hash(key)
        slot = self.table[index]
        for i, (k, v) in enumerate(slot):
            if k == key:
                del slot[i]
                return
        raise KeyError("Key not found")
```

**Using the Hashtable:**  
Now that we have our Hashtable class, let's see how we can use it:
```python
# Create a new hashtable
ht = Hashtable(10)

# Insert key-value pairs
ht.insert("apple", 10)
ht.insert("banana", 5)
ht.insert("orange", 15)

# Retrieve values
print(ht.get("apple"))  # Output: 10
print(ht.get("banana"))  # Output: 5
print(ht.get("orange"))  # Output: 15

# Update a value
ht.insert("apple", 20)
print(ht.get("apple"))  # Output: 20

# Delete a key-value pair
ht.delete("banana")

# Try to retrieve the deleted key
print(ht.get("banana"))  # Raises KeyError: Key not found
```


**Advantages of Hashtables:**  
**Efficient Lookup:** Hashtables provide constant-time lookup for key-value pairs, making them ideal for scenarios that require fast data retrieval.  
**Fast Insertion and Deletion:** Inserting and deleting elements in a hashtable typically have an average time complexity of O(1), making these operations efficient.  
**Flexibility:** Hashtables can handle a large amount of data and accommodate dynamic resizing without significantly impacting performance.  

**Disadvantages of Hashtables:**  
**Collision Handling:** Hash collisions occur when two different keys generate the same hash value. Resolving collisions requires additional operations, such as chaining or open addressing, which can affect performance.  
**Memory Overhead:** Hashtables consume additional memory to maintain the hash table structure and handle collisions. This overhead can be significant when storing a large number of elements.  
**Lack of Ordering:** Hashtables do not preserve the order of elements, which can be a disadvantage when ordering is important in a specific use case.  

**Time Complexity:**  
**Average case:**  
Insertion: O(1)
Deletion: O(1)
Lookup: O(1)

**Worst case:**  
Insertion: O(n)
Deletion: O(n)
Lookup: O(n)

Space Complexity: O(n), where n is the number of elements stored in the hashtable.