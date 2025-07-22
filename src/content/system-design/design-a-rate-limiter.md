---
title: "Design A Rate Limiter"
description: "Learn to design a rate limiter to control traffic in network systems, preventing abuse and ensuring service stability."
date: "2023-01-05"
order: 5
---

# Design A Rate Limiter

In network systems, a rate limiter controls the rate of traffic sent by clients or services. In the context of HTTP, a rate limiter restricts the number of client requests within a specified period. Excess requests are blocked once the defined threshold is exceeded. Examples include:

*   A user can create no more than 2 posts per second.
*   A maximum of 10 accounts can be created per day from the same IP address.
*   Rewards can be claimed no more than 5 times per week from the same device.

Before diving into the design, let's explore the benefits of an API rate limiter:

*   **Prevent DoS Attacks:** Rate limiting helps prevent resource exhaustion caused by Denial of Service (DoS) attacks, whether intentional or unintentional. Most large tech companies enforce rate limiting on their APIs (e.g., Twitter limits tweets, Google Docs APIs have read request limits). Blocking excess calls safeguards resources.
*   **Reduce Cost:** Limiting requests reduces operational costs, especially for companies using paid third-party APIs (e.g., credit checks, payment processing, health record retrieval). Efficiently managing API calls is crucial for cost control.
*   **Prevent Server Overload:** Rate limiters filter out excessive requests from bots or misbehaving users, reducing the load on servers and maintaining system stability.

## Step 1: Understand the Problem and Establish Design Scope

Rate limiting can be implemented using various algorithms, each with its strengths and weaknesses. Clarifying the type of rate limiter we are building is essential.

**Clarification Questions & Assumptions:**
*   **Type of Rate Limiter:** Server-side API rate limiter.
*   **Throttling Basis:** Flexible enough to support various throttle rules (e.g., based on IP, user ID).
*   **System Scale:** Must handle a large number of requests.
*   **Distributed Environment:** Yes, the system will work in a distributed environment.
*   **Implementation:** A design decision (separate service or in application code).
*   **User Notification:** Yes, users should be informed when their requests are throttled.

**Requirements Summary:**
*   Accurately limit excessive requests.
*   Low latency: Should not slow down HTTP response time.
*   Memory efficiency: Use as little memory as possible.
*   Distributed rate limiting: Shareable across multiple servers or processes.
*   Exception handling: Provide clear exceptions to throttled users.
*   High fault tolerance: The rate limiter should not be a single point of failure.

## Step 2: Propose High-Level Design and Get Buy-in

We will use a basic client-server communication model.

### Where to Place the Rate Limiter?

Rate limiting can be implemented client-side or server-side. Client-side implementation is generally unreliable due to the ease of forging requests and lack of control over client logic. Therefore, we focus on **server-side implementation**.

An alternative to placing the rate limiter directly on API servers is to introduce a **rate limiter middleware**.

If a client sends 3 requests to an API that allows 2 requests per second, the first two requests are processed, while the third is throttled by the middleware, returning an HTTP 429 status code (Too Many Requests).

In modern microservices architectures, rate limiting is often integrated into an **API Gateway**. An API Gateway is a managed service that supports features like rate limiting, SSL termination, authentication, and IP whitelisting.

Choosing where to implement the rate limiter depends on your company's technology stack, resources, and goals:

*   **Technology Stack:** Evaluate if your current programming language and cache service are efficient for server-side rate limiting.
*   **Algorithm Control:** Implementing it server-side offers full control over the algorithm, while third-party gateways might limit choices.
*   **Existing Infrastructure:** If you already use an API Gateway for other functionalities, extending it with rate limiting might be logical.
*   **Engineering Resources:** Building a custom rate limiting service requires significant resources; commercial API gateways offer a faster solution if resources are limited.

### Algorithms for Rate Limiting

Various algorithms can implement rate limiting, each with pros and cons. Understanding them helps in choosing the right one.

*   Token Bucket
*   Leaking Bucket
*   Fixed Window Counter
*   Sliding Window Log
*   Sliding Window Counter

#### Token Bucket Algorithm

Widely used and well-understood (e.g., by Amazon and Stripe). It works as follows:

*   A token bucket has a predefined capacity. Tokens are added to the bucket at a preset refill rate. Once full, no more tokens are added.
*   Each request consumes one token. If enough tokens are available, the request proceeds and a token is consumed. If not, the request is dropped.

**Parameters:**
*   **Bucket size:** Maximum number of tokens allowed in the bucket.
*   **Refill rate:** Number of tokens added to the bucket per unit of time (e.g., per second).

The number of buckets needed depends on the rate-limiting rules. For example, different buckets might be needed for different API endpoints, per IP address, or a global bucket for all requests.

**Pros:**
*   Easy to implement.
*   Memory efficient.
*   Allows bursts of traffic for short periods as long as tokens are available.

**Cons:**
*   Challenging to tune bucket size and refill rate properly.

#### Leaking Bucket Algorithm

Similar to token bucket but processes requests at a fixed rate, often implemented with a FIFO queue. The algorithm works as follows:

*   When a request arrives, if the queue is not full, the request is added.
*   Otherwise, the request is dropped.
*   Requests are pulled from the queue and processed at regular intervals.

**Parameters:**
*   **Bucket size:** Equal to the queue size, holding requests to be processed at a fixed rate.
*   **Outflow rate:** Defines how many requests are processed at a fixed rate.

Shopify uses leaky buckets for rate limiting.

**Pros:**
*   Memory efficient due to limited queue size.
*   Suitable for use cases requiring a stable outflow rate.

**Cons:**
*   A burst of traffic can fill the queue with older requests, leading to recent requests being rate-limited if not processed in time.
*   Tuning parameters can be challenging.

#### Fixed Window Counter Algorithm

This algorithm divides the timeline into fixed-sized windows and assigns a counter to each window. It works as follows:

*   Each request increments the counter.
*   Once the counter reaches a predefined threshold, new requests are dropped until a new time window begins.

**Problem:** A burst of traffic at the edges of time windows can allow more requests than the allowed quota. For example, with a limit of 5 requests per minute, a client could make 5 requests at 2:00:59 and 5 more at 2:01:01, effectively sending 10 requests within a very short period across two windows.

**Pros:**
*   Memory efficient.
*   Easy to understand.
*   Quota resetting at unit time windows fits certain use cases.

**Cons:**
*   Traffic spikes at window edges can allow more requests than intended.

#### Sliding Window Log Algorithm

This algorithm addresses the edge-case issue of the fixed window counter. It works as follows:

*   Keeps track of request timestamps, typically in a cache like Redis sorted sets.
*   When a new request arrives, all outdated timestamps (older than the start of the current time window) are removed.
*   If the number of remaining timestamps plus the new request exceeds the limit, the request is throttled.
*   Otherwise, the request is allowed, and its timestamp is added.

**Pros:**
*   Highly accurate control over the request rate, as it doesn't suffer from the edge-case problem of fixed window counters.

**Cons:**
*   Memory intensive, as it stores a timestamp for every request.

#### Sliding Window Counter Algorithm

This is a hybrid approach that combines the fixed window counter and sliding window log to provide a more accurate and memory-efficient solution than the fixed window counter.

It works by calculating a weighted average of the current window's count and the previous window's count. For example, to check the rate for the current minute, it would take the count from the previous minute's window and a fraction of the current minute's window, based on how much of the current window has passed.

**Pros:**
*   Offers a good balance between accuracy and memory efficiency.
*   Smooths out traffic spikes better than the fixed window counter.

**Cons:**
*   More complex to implement than fixed window counter.

### High-Level Architecture

Let's consider a generic high-level architecture for a distributed rate limiter:

*   **Client:** Sends requests to the service.
*   **Rate Limiter Service:** Intercepts requests. It queries a data store to check the current rate limit status for a given client/key. If allowed, the request proceeds; otherwise, it's rejected.
*   **Data Store (e.g., Redis):** A fast, distributed key-value store suitable for storing counters and timestamps for rate limiting. Its in-memory nature provides low latency, and its persistence features ensure durability.

When a request comes in, the rate limiter middleware/service extracts a unique identifier (e.g., IP address, user ID, API key). It then interacts with a distributed data store (like Redis) to increment counters or add timestamps based on the chosen algorithm. If the limit is exceeded, a 429 Too Many Requests response is returned.

### Deep Dive: Components and Considerations

**1. Rules Engine:**

This component defines and manages the rate limiting rules. Rules can be configured based on various parameters:

*   **User ID:** e.g., 100 requests/minute per user.
*   **IP Address:** e.g., 500 requests/minute per IP.
*   **API Endpoint:** e.g., `/api/v1/create_post` allows 2 requests/second, while `/api/v1/get_data` allows 100 requests/second.
*   **HTTP Method:** e.g., POST requests might have stricter limits than GET requests.
*   **Headers:** Custom headers can also define rules.

Rules are typically stored in a configuration service or database and loaded by the rate limiter service. They should be dynamically updatable.

**2. Counters/Timestamps Storage:**

A distributed, highly available, and low-latency data store is crucial. Redis is an excellent choice due to its in-memory nature and support for various data structures (e.g., sorted sets for sliding window log, simple integers for counters). Each rate limiting rule would correspond to a key in Redis.

**3. Notifications:**

When a request is throttled, the system should inform the user. This is typically done by returning an HTTP 429 (Too Many Requests) status code. Additionally, the response headers can include:

*   `X-RateLimit-Limit`: The maximum number of requests allowed in the current window.
*   `X-RateLimit-Remaining`: The number of requests remaining in the current window.
*   `X-RateLimit-Reset`: The time (e.g., in UTC epoch seconds) when the current rate limit window resets.

This provides transparency and allows clients to adjust their request rates.

**4. Monitoring and Alerting:**

Comprehensive monitoring is essential to track:

*   **Throttled requests:** Number of requests blocked by the rate limiter.
*   **Rate limit breaches:** When specific limits are hit.
*   **System health:** Latency, error rates of the rate limiter service itself.

Alerts should be configured for critical events to ensure prompt response to issues.

**5. Fault Tolerance and Consistency:**

*   **High Availability:** The rate limiter service should be highly available. Deploying it across multiple instances behind a load balancer ensures continued operation even if one instance fails.
*   **Data Consistency:** For distributed rate limiters, eventual consistency for counters might be acceptable for some use cases, while strong consistency is preferred for others (e.g., financial transactions). Redis replication and clustering can help with consistency and availability.
*   **Fallback Mechanism:** If the rate limiter service itself fails or becomes unreachable, a fallback mechanism should be in place (e.g., temporarily allowing all requests or a default lower limit) to prevent the rate limiter from becoming a single point of failure that brings down the entire system.

### Step 4: Wrap Up (Additional Considerations)

*   **Hard vs. Soft Throttling:** Differentiate between hard limits (strict blocking) and soft limits (e.g., delaying requests, or giving a warning).
*   **Client-side Caching:** Clients can also implement some form of caching of their rate limit status to reduce unnecessary requests to the rate limiter.
*   **Granularity:** Decide on the granularity of rate limiting (e.g., per second, per minute, per hour, per day).
*   **Bursts:** How to handle legitimate bursts of traffic vs. malicious spikes. Token bucket is good for bursts.
*   **Distributed Counters:** Challenges of maintaining accurate counters in a distributed system, especially with network latency and clock skew.
*   **Security:** Ensure the rate limiter itself is not vulnerable to attacks.
*   **A/B Testing:** Experiment with different rate limiting algorithms and thresholds to find the optimal balance for user experience and system protection.
*   **Graceful Degradation:** If the system is under extreme load, the rate limiter can prioritize critical requests and degrade gracefully for less critical ones.

Designing a robust and scalable rate limiter involves careful consideration of algorithms, placement, data storage, and operational aspects to protect your services from abuse and ensure stability. 