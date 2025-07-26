---
title: "Hashmaps"
description: "A guide to implementing and utilizing the Hashmap data structure, along with an analysis of its time complexity..."
pubDate: "Jun 1 2023"
heroImage: "/post_img.webp"
---
Hashmaps, also known as dictionaries or associative arrays, are a fundamental data structure in computer science. They provide fast access to data by associating keys with values using a hash function. In this blog post, we will explore the concept of hashmaps, their advantages, and how to implement a hashmap in Python.

**Understanding Hashmaps:**  
A hashmap is a collection of key-value pairs, where each key is unique. It utilizes a hash function to map keys to specific indexes in an underlying array, allowing for constant-time retrieval, insertion, and deletion operations. This makes hashmaps a powerful tool for solving a wide range of problems efficiently.

**Implementing a Hashmap in Python:**  
Let's dive into the implementation of a hashmap in Python. We'll start by defining a HashMap class that encapsulates the necessary functionality.
```python
class HashMap:
    def __init__(self):
        self.size = 16  # Initial size of the underlying array
        self.buckets = [[] for _ in range(self.size)]  # List of buckets, each containing key-value pairs

    def _hash(self, key):
        return hash(key) % self.size  # Hash function to determine the index of a key in the array

    def put(self, key, value):
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)  # Update the value if the key already exists
                return

        bucket.append((key, value))  # Add the key-value pair to the bucket

    def get(self, key):
        index = self._hash(key)
        bucket = self.buckets[index]

        for existing_key, value in bucket:
            if existing_key == key:
                return value

        raise KeyError(f"Key '{key}' not found")

    def remove(self, key):
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]  # Remove the key-value pair from the bucket
                return

        raise KeyError(f"Key '{key}' not found")
```

**Using the Hashtable:**  
Now that we have our Hashtable class, let's see how we can use it:
```python
# Creating a hashmap
hashmap = {}

# Adding key-value pairs to the hashmap
hashmap["apple"] = 5
hashmap["banana"] = 10
hashmap["orange"] = 7

# Accessing values by key
print(hashmap["apple"])  # Output: 5

# Updating a value
hashmap["banana"] = 15

# Removing a key-value pair
del hashmap["orange"]

# Checking if a key exists
if "banana" in hashmap:
    print("Key 'banana' exists in the hashmap")

# Getting the number of key-value pairs
size = len(hashmap)
print("Size of hashmap:", size)

# Iterating over keys
for key in hashmap:
    print(key, "->", hashmap[key])
```

**Advantages of Hashmaps:**  
**Fast Retrieval:** Hashmaps provide constant-time access to values based on their keys, regardless of the size of the dataset.  
**Efficient Insertion and Deletion:** Hashmaps offer efficient insertion and deletion operations, making them suitable for dynamic data structures.  
**Flexible Key Types:** Hashmaps can handle various types of keys, including strings, integers, and custom objects.  
**Key-Value Association:** Hashmaps allow for easy association of values with their corresponding keys, enabling intuitive data organization.  
**Widely Used:** Hashmaps are extensively used in various domains, including databases, caching mechanisms, language implementations, and algorithm design.  

**Disadvantages of Hashmaps:**  
**Hash Collisions:** Hashmaps may encounter collisions when different keys produce the same hash value, requiring additional handling and potentially impacting performance.  
**Memory Overhead:** Hashmaps consume additional memory to store the underlying array and handle potential collisions, which can be a concern for large datasets.  
**Order Unpredictability:** Hashmaps do not guarantee a specific order of iteration over keys, which may not be desirable in certain use cases.  

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