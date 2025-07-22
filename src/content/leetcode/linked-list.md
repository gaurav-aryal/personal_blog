---
title: Linked List
description: Java solutions with explanations, time and space complexity for Linked List problems.
date: "June 1 2025"
---

# Linked List Pattern

Linked Lists are fundamental data structures that consist of nodes connected by pointers. They're particularly useful for:
- Dynamic memory allocation
- Efficient insertions and deletions
- Implementing other data structures
- Memory-efficient storage
- Circular data structures

## 1. Reverse Linked List (Easy)

**Problem:** Given the head of a singly linked list, reverse the list, and return the reversed list.

**Solution:**
```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        
        return prev;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(1)

## 2. Merge Two Sorted Lists (Easy)

**Problem:** Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.

**Solution:**
```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        
        curr.next = (l1 != null) ? l1 : l2;
        return dummy.next;
    }
}
```

**Time Complexity:** O(n + m) where n and m are lengths of input lists
**Space Complexity:** O(1)

## 3. Reorder List (Medium)

**Problem:** Given the head of a singly linked list L: L0 → L1 → … → Ln-1 → Ln, reorder it to: L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …

**Solution:**
```java
class Solution {
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;
        
        // Find middle
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // Reverse second half
        ListNode prev = null, curr = slow;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        
        // Merge two halves
        ListNode first = head, second = prev;
        while (second.next != null) {
            ListNode temp = first.next;
            first.next = second;
            first = temp;
            
            temp = second.next;
            second.next = first;
            second = temp;
        }
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(1)

## 4. Remove Nth Node From End of List (Medium)

**Problem:** Given the head of a linked list, remove the nth node from the end of the list and return its head.

**Solution:**
```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode first = dummy;
        ListNode second = dummy;
        
        // Move first pointer n+1 steps ahead
        for (int i = 0; i <= n; i++) {
            first = first.next;
        }
        
        // Move both pointers until first reaches end
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        
        // Remove the nth node
        second.next = second.next.next;
        return dummy.next;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(1)

## 5. Copy List with Random Pointer (Medium)

**Problem:** A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null. Construct a deep copy of the list.

**Solution:**
```java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        
        // Step 1: Create copy nodes next to original nodes
        Node curr = head;
        while (curr != null) {
            Node copy = new Node(curr.val);
            copy.next = curr.next;
            curr.next = copy;
            curr = copy.next;
        }
        
        // Step 2: Set random pointers
        curr = head;
        while (curr != null) {
            if (curr.random != null) {
                curr.next.random = curr.random.next;
            }
            curr = curr.next.next;
        }
        
        // Step 3: Separate original and copy lists
        Node dummy = new Node(0);
        Node copyCurr = dummy;
        curr = head;
        
        while (curr != null) {
            copyCurr.next = curr.next;
            curr.next = curr.next.next;
            curr = curr.next;
            copyCurr = copyCurr.next;
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(1)

## 6. Add Two Numbers (Medium)

**Problem:** You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

**Solution:**
```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        int carry = 0;
        
        while (l1 != null || l2 != null || carry != 0) {
            int sum = carry;
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(max(n,m)) where n and m are lengths of input lists
**Space Complexity:** O(max(n,m))

## 7. Linked List Cycle (Easy)

**Problem:** Given head, the head of a linked list, determine if the linked list has a cycle in it.

**Solution:**
```java
class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            
            if (slow == fast) return true;
        }
        
        return false;
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(1)

## 8. Find the Duplicate Number (Medium)

**Problem:** Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive, find the duplicate number.

**Solution:**
```java
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];
        
        // Find meeting point
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        
        // Find entrance to cycle
        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        
        return slow;
    }
}
```

**Time Complexity:** O(n) where n is the length of the array
**Space Complexity:** O(1)

## 9. LRU Cache (Medium)

**Problem:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

**Solution:**
```java
class LRUCache {
    private class Node {
        int key, value;
        Node prev, next;
        Node(int k, int v) {
            key = k;
            value = v;
        }
    }
    
    private Map<Integer, Node> cache;
    private Node head, tail;
    private int capacity;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        cache = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            remove(node);
            add(node);
            return node.value;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            remove(cache.get(key));
        }
        if (cache.size() == capacity) {
            remove(tail.prev);
        }
        add(new Node(key, value));
    }
    
    private void add(Node node) {
        cache.put(node.key, node);
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
    
    private void remove(Node node) {
        cache.remove(node.key);
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
}
```

**Time Complexity:** O(1) for both get and put operations
**Space Complexity:** O(capacity)

## 10. Merge K Sorted Lists (Hard)

**Problem:** You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

**Solution:**
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            curr.next = node;
            curr = curr.next;
            
            if (node.next != null) {
                pq.offer(node.next);
            }
        }
        
        return dummy.next;
    }
}
```

**Time Complexity:** O(n log k) where n is total number of nodes and k is number of lists
**Space Complexity:** O(k) for the priority queue

## 11. Reverse Nodes in K Group (Hard)

**Problem:** Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

**Solution:**
```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        while (head != null) {
            ListNode tail = prev;
            // Check if there are k nodes left
            for (int i = 0; i < k; i++) {
                tail = tail.next;
                if (tail == null) return dummy.next;
            }
            
            ListNode next = tail.next;
            ListNode[] reversed = reverse(head, tail);
            head = reversed[0];
            tail = reversed[1];
            
            // Connect the reversed group
            prev.next = head;
            tail.next = next;
            prev = tail;
            head = next;
        }
        
        return dummy.next;
    }
    
    private ListNode[] reverse(ListNode head, ListNode tail) {
        ListNode prev = tail.next;
        ListNode curr = head;
        
        while (prev != tail) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        
        return new ListNode[]{tail, head};
    }
}
```

**Time Complexity:** O(n) where n is the number of nodes
**Space Complexity:** O(1)

## Key Takeaways

1. Linked List is perfect for:
   - Dynamic memory allocation
   - Efficient insertions and deletions
   - Implementing other data structures
   - Memory-efficient storage
   - Circular data structures

2. Common patterns:
   - Two pointer technique (fast & slow)
   - Dummy node for edge cases
   - Reversing linked lists
   - Merging sorted lists
   - Cycle detection

3. Tips:
   - Always check for null pointers
   - Use dummy nodes to handle edge cases
   - Consider using two pointers for complex operations
   - Think about space-time tradeoffs
   - Consider using sentinel nodes 