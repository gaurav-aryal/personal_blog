---
title: "Understanding Time Complexity and Space Complexity"
description: "Explains the concepts of time complexity and space complexity in computer science with a few examples..."
pubDate: "Apr 17 2023"
heroImage: "/post_img.webp"
---
In computer science, time complexity and space complexity are two important concepts that help in analyzing the efficiency of algorithms. Time complexity refers to the amount of time an algorithm takes to run, while space complexity refers to the amount of memory required by an algorithm to solve a problem. In this blog post, we will explore both concepts in detail and provide Python code examples to help illustrate these concepts.

**Time Complexity**  
Time complexity is a measure of the amount of time an algorithm takes to run as a function of the input size. This is typically expressed using Big O notation, which provides an upper bound on the running time of the algorithm. The Big O notation uses a function to describe the upper limit of the running time, which can be written as O(f(n)), where f(n) is a function of the input size n.  
Here are some examples of time complexities and their corresponding functions:

O(1) - constant time complexity, where the running time of the algorithm is constant and independent of the input size.
<pre><code class="language-python">
def example_constant_time():
    a = 5
    b = 10
    c = a + b
    print(c)
</code></pre>  
O(n) - linear time complexity, where the running time of the algorithm increases linearly with the input size.  
<pre><code class="language-python">
def example_linear_time(n):
    for i in range(n):
        print(i)
</code></pre>  
O(n^2) - quadratic time complexity, where the running time of the algorithm increases exponentially with the input size.  
<pre><code class="language-python">
def example_quadratic_time(n):
    for i in range(n):
        for j in range(n):
            print(i, j)
</code></pre>  
It's important to note that the time complexity is an upper bound on the running time, and the actual running time can be better or worse than the Big O notation suggests.

**Space Complexity**  
Space complexity is a measure of the amount of memory required by an algorithm to solve a problem. It is also typically expressed using Big O notation, which provides an upper bound on the memory required by the algorithm. The Big O notation uses a function to describe the upper limit of the memory usage, which can be written as O(f(n)), where f(n) is a function of the input size n.  
Here are some examples of space complexities and their corresponding functions:  
O(1) - constant space complexity, where the amount of memory required by the algorithm is constant and independent of the input size.  
<pre><code class="language-python">
def example_constant_space(n):
    a = 5
    b = 10
    c = a + b
    print(c)
</code></pre>  
O(n) - linear space complexity, where the amount of memory required by the algorithm increases linearly with the input size.  
<pre><code class="language-python">
def example_linear_space(n):
    array = [0] * n
    for i in range(n):
        array[i] = i
    print(array)
</code></pre>  
O(n^2) - quadratic space complexity, where the amount of memory required by the algorithm increases exponentially with the input size.  
<pre><code class="language-python">
def example_quadratic_space(n):
    matrix = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = i * j
    print(matrix)
</code></pre>
Just like time complexity, space complexity is also an upper bound on the memory usage, and the actual memory usage can be better or worse than the Big O notation suggests.