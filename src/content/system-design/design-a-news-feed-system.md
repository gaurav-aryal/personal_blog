---
title: "Design A News Feed System"
description: "Design a scalable news feed system, focusing on feed publishing, aggregation, and key architectural components for high traffic."
date: "2023-01-12"
order: 12
---

## Design A News Feed System

Designing a news feed system is a common system design interview question, with challenges similar to building a chat system or social network. The core task involves efficiently delivering real-time updates to users from various sources. This chapter explores the design of a scalable news feed system.

### Step 1: Understand the Problem and Establish Design Scope

A news feed system allows users to see a stream of content (posts, activities) from people or entities they follow. Clarifying requirements is crucial for a well-designed system.

**Clarification Questions & Assumptions:**
*   **Feed Content:** Can include text, images, and videos.
*   **Feed Sorting:** Initially, news feed is sorted in reverse chronological order. More complex ranking can be considered later.
*   **User Relationships:** Users can follow other users.
*   **Number of Followers:** A user can have many followers (e.g., 5,000 friends).
*   **Traffic Volume:** High daily active users (e.g., 10 million DAU).
*   **System Type:** Mobile app, web app, or both.

**Key Characteristics of a News Feed System:**
*   **High availability:** The feed should almost always be accessible.
*   **Scalability:** Efficiently handle a large number of users and posts.
*   **Low latency:** Users should see new posts with minimal delay.
*   **Fault tolerance:** The system should remain operational even if some components fail.

**Back-of-the-Envelope Estimation:**
(Assuming 10 million DAU, 5000 friends/user, 1 post/day/user)

*   **Posts per second (write QPS):** 10M users * 1 post/day / (24*3600 seconds/day) ≈ 115 posts/second.
*   **Feed reads per second (read QPS):** Each DAU refreshes feed 5 times a day: 10M users * 5 reads/day / (24*3600 seconds/day) ≈ 578 reads/second. This will be much higher considering millions of users constantly refreshing their feed.
*   **Storage:** Storing posts, user data, and relationship data. Media storage (images/videos) can be significant.

### Step 2: Propose High-Level Design and Get Buy-in

The news feed system can be broken down into two primary flows: **Feed Publishing** (writing a post) and **News Feed Building/Retrieval** (reading a feed).

#### Feed Publishing Flow (Write Path)

When a user creates a new post:
1.  The client (mobile app or web) sends a request to the API server.
2.  The API server validates the request and stores the post content in a database (e.g., a NoSQL database for flexible schemas).
3.  The post is then fanned out to the news feeds of all followers.

**Fanout Types:**
*   **Fanout-on-Write (Push Model):** When a post is created, it is immediately pushed to the inboxes (news feeds) of all followers. This is common for systems where reads are much more frequent than writes (e.g., Twitter). Pros: Fast feed retrieval. Cons: High write amplification, potential for hot spots for users with many followers.
*   **Fanout-on-Read (Pull Model):** When a user requests their news feed, the system pulls posts from all the people they follow, aggregates them, sorts them, and then displays the feed. This is common for systems where reads are less frequent or the number of followers is small (e.g., Facebook, where users have many friends but fewer active followers than a celebrity on Twitter). Pros: Efficient writes, less storage. Cons: Slower feed retrieval, heavy read load on the database.

Most large-scale news feed systems use a **hybrid approach**, leveraging both push and pull models to optimize performance and resource usage. For instance, posts from highly followed users might use a pull model, while regular users use a push model.

#### News Feed Building/Retrieval Flow (Read Path)

When a user requests their news feed:
1.  The client sends a request to the API server.
2.  The API server retrieves the personalized news feed.
3.  The aggregated and sorted feed is returned to the client.

This involves querying multiple data sources, merging the results, and sorting them. Caching plays a critical role here to achieve low latency.

### Step 3: Design Deep Dive

Let's delve deeper into key components and considerations.

#### Data Models

*   **User Table:** Stores user information (ID, name, profile picture, etc.).
*   **Post Table:** Stores post content (post ID, user ID, text, media URLs, timestamp).
*   **Follower/Following Table:** Stores relationships (follower ID, followed ID).
*   **News Feed Table (for Fanout-on-Write):** Stores personalized feeds for each user (user ID, post ID, timestamp).

#### Storage Choices

*   **Posts & User Data:** A NoSQL database (e.g., Cassandra, DynamoDB) is suitable for its scalability, high write throughput (for posts), and ability to handle large volumes of unstructured data. For relationships, a graph database could be considered, but a relational database or NoSQL with appropriate indexing can also work.
*   **News Feed (User Inboxes):** A low-latency, high-throughput key-value store or in-memory cache (e.g., Redis, Memcached) is ideal for storing materialized news feeds for fast retrieval. This acts as a cache for the fanout-on-write model.

#### Feed Publishing Deep Dive

1.  **Post Creation:** User creates a post, sent to a dedicated Write API service.
2.  **Write to Database:** The post is stored in the Post Table.
3.  **Asynchronous Fanout (for Push Model):**
    *   A message queue (e.g., Kafka, RabbitMQ) is used to asynchronously fan out the post to followers. The Write API sends a message to the queue.
    *   **Fanout Workers:** Consume messages from the queue. For each follower of the post creator, the worker inserts the post into the follower's news feed inbox (in the News Feed Table/Cache).
    *   **Dealing with Super-Followers:** For users with millions of followers (e.g., celebrities), pushing to all inboxes can be extremely write-intensive. A hybrid approach would push to a subset of active followers and use a pull mechanism for the rest or for highly engaged users.

#### News Feed Retrieval Deep Dive

1.  **Feed Request:** User requests their feed from a Read API service.
2.  **Cache Check:** The Read API first checks a personalized feed cache (e.g., Redis) for the user's news feed.
    *   If found, return directly (low latency).
3.  **Database Fallback/Aggregation (for Pull Model/Cache Misses):** If not in cache or for hybrid models:
    *   Query the Following Table to get a list of followed users.
    *   Query the Post Table for recent posts from these followed users.
    *   Aggregate and sort the posts (e.g., by timestamp).
    *   Cache the generated feed for future requests.

#### Other Considerations

*   **Caching Strategy:** Cache frequently accessed feeds and popular posts. Implement cache invalidation or time-to-live (TTL) policies.
*   **Load Balancing:** Distribute traffic across API servers and other components.
*   **Content Delivery Network (CDN):** Serve static media content (images, videos) from edge locations for lower latency.
*   **Real-time Updates:** For push models, a WebSockets-based notification service can inform clients about new posts, prompting them to refresh their feeds or receiving direct updates.
*   **Ranking/Personalization:** Beyond chronological order, a ranking algorithm can prioritize posts based on relevance, user engagement, or other factors (e.g., based on machine learning models). This adds significant complexity.
*   **Error Handling and Retries:** Implement robust mechanisms for dealing with failures in writes, fanouts, and reads.
*   **Monitoring and Analytics:** Track system performance, user engagement, feed latency, and other metrics to identify bottlenecks and improve the system.

### Step 4: Wrap Up (Additional Considerations)

*   **Scalability Challenges:** Discuss horizontal scaling of all stateless components (API servers, fanout workers). Database scaling (sharding, replication) is crucial.
*   **Consistency vs. Latency:** News feeds often prioritize eventual consistency for write operations to achieve lower latency for reads.
*   **Offline Support:** How can the system support users viewing feeds while offline?
*   **Spam and Abuse:** Mechanisms to detect and filter out spam or inappropriate content.
*   **Data Archiving:** Strategy for handling old posts or media that are less frequently accessed.

Designing a news feed system involves balancing real-time delivery with scalability and consistency, making thoughtful choices about fanout mechanisms, data storage, and caching strategies. 