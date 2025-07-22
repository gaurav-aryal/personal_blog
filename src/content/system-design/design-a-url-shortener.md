---
title: "Design A URL Shortener"
description: "A classic system design interview question: designing a URL shortening service like TinyURL."
date: "2023-01-09"
order: 9
---

### Step 1: Understanding the Problem and Design Scope

Key clarification questions and assumptions:

*   **Example:** A long URL (e.g., `https://www.example.com/long-url`) is transformed into a short alias (e.g., `https://tinyurl.com/y7keocwj`). Clicking the alias redirects to the original URL.
*   **Traffic Volume:** Approximately 100 million URLs generated per day.
*   **Shortened URL Length:** As short as possible.
*   **Allowed Characters:** Numbers (0-9) and characters (a-z, A-Z).
*   **Deletion/Updates:** For simplicity, shortened URLs cannot be deleted or updated.

**Basic Use Cases:**
1.  **URL Shortening:** Input a long URL, receive a shorter URL.
2.  **URL Redirecting:** Input a short URL, redirect to the original long URL.
3.  **Non-functional requirements:** High availability, scalability, and fault tolerance.

**Back-of-the-Envelope Estimation:**
*   **Write Operations:** 100 million URLs/day ≈ 1160 writes/second.
*   **Read Operations:** Assuming a 10:1 read-to-write ratio, ≈ 11,600 reads/second.
*   **Total Records (10 years):** 365 billion records.
*   **Average URL Length:** 100 bytes.
*   **Storage Requirement (10 years):** 36.5 TB.

### Step 2: High-Level Design

**API Endpoints (REST-style):**
1.  **URL Shortening:**
    *   `POST api/v1/data/shorten`
    *   Request: `{longUrl: longURLString}`
    *   Returns: `shortURL`
2.  **URL Redirecting:**
    *   `GET api/v1/shortUrl`
    *   Returns: `longURL` for HTTP redirection.

**URL Redirecting Flow:**
When a short URL is accessed, the server receives the request and redirects to the original long URL. This involves a **301 redirect** (permanent, cached by browser) or **302 redirect** (temporary, subsequent requests hit the service). 301 reduces server load, while 302 is better for analytics.

**(Figure 1: Illustration of URL redirecting process)**
**(Figure 2: Detailed communication between client and server for redirection)**

**URL Shortening Flow:**
Assumes the short URL format: `www.tinyurl.com/{hashValue}`. A hash function `fx` maps a long URL to a unique `hashValue`.

**(Figure 3: Long URL to Hash Value mapping)**

### Step 3: Design Deep Dive

**Data Model:**
Instead of an in-memory hash table, a relational database is preferred to store `<shortURL, longURL>` mappings.

**(Figure 4: Simplified database table design with id, shortURL, longURL columns)**

**Hash Function:**
The `hashValue` consists of 62 possible characters (0-9, a-z, A-Z). To support 365 billion URLs, the length of the `hashValue` needs to be 7 characters (since 62^7 ≈ 3.5 trillion, which is sufficient).

**Two approaches for Hash Functions:**

1.  **Hash + Collision Resolution:**
    *   Use standard hash functions (CRC32, MD5, SHA-1). These produce longer hashes.
    *   Take the first 7 characters, but this can cause collisions.
    *   Collision resolution: Recursively append a predefined string until no collision is found. Bloom filters can optimize collision checking.
    
    **(Figure 5: Collision resolution process)**

2.  **Base 62 Conversion (Preferred Approach):**
    *   Convert a unique numerical ID (from a unique ID generator) into a base 62 representation.
    *   Example: 11157 (base 10) converts to "2TX" (base 62).
    
    **(Figure 6: Base 62 conversion process example)**

**Comparison of Approaches:**
| Feature              | Hash + Collision Resolution      | Base 62 Conversion               |
| :------------------- | :------------------------------- | :------------------------------- |
| Short URL Length     | Fixed                            | Not fixed, depends on ID        |
| Unique ID Generator  | Not needed                       | Depends on it                    |
| Collision            | Possible, needs resolution       | Not possible (ID is unique)      |
| Next Short URL       | Not predictable                  | Predictable (if ID increments)   |

**URL Shortening Deep Dive (using Base 62 Conversion):**

**(Figure 7: URL shortening flow)**

1.  **Input:** `longURL`.
2.  **Check Database:** See if `longURL` already exists.
3.  **Return Existing:** If found, return the existing `shortURL`.
4.  **Generate New:** If new, generate a unique `ID` (e.g., `2009215674938`) from a distributed unique ID generator.
5.  **Convert ID:** Convert the `ID` to `shortURL` using base 62 conversion (e.g., "zn9edcu").
6.  **Save to DB:** Store `ID`, `shortURL`, and `longURL` in the database.

**URL Redirecting Deep Dive:**

**(Figure 8: Detailed URL redirecting design with caching)**

1.  **User Clicks:** User clicks a `shortURL`.
2.  **Load Balancer:** Forwards request to web servers.
3.  **Cache Check:** If `shortURL` is in cache, return `longURL` directly.
4.  **Database Fetch:** If not in cache, fetch `longURL` from the database. Handle invalid `shortURL`s.
5.  **Return LongURL:** Return the `longURL` to the user.

### Step 4: Wrap Up (Additional Considerations)

*   **Rate Limiter:** Prevent malicious users from overwhelming the service with requests.
*   **Web Server Scaling:** Web tier is stateless, easy to scale by adding/removing servers.
*   **Database Scaling:** Use replication and sharding.
*   **Analytics:** Integrate solutions to track click rates and other metrics.
*   **Availability, Consistency, Reliability:** Core principles for large systems. 