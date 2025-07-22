---
title: "Back-of-the-envelope Estimation"
description: "Learn to estimate system capacity and performance requirements in system design interviews using back-of-the-envelope calculations."
date: "2023-01-03"
order: 3
---

## Power of Two for Data Volume

Understanding data volume units based on powers of two is crucial for distributed systems. A byte consists of 8 bits. An ASCII character typically uses one byte. The table below illustrates common data volume units:

| Power | Approximate Value | Full Name   | Short Name |
|-------|-------------------|-------------|------------|
| 10    | 1 Thousand        | 1 Kilobyte  | 1 KB       |
| 20    | 1 Million         | 1 Megabyte  | 1 MB       |
| 30    | 1 Billion         | 1 Gigabyte  | 1 GB       |
| 40    | 1 Trillion        | 1 Terabyte  | 1 TB       |
| 50    | 1 Quadrillion     | 1 Petabyte  | 1 PB       |

## Latency Numbers for Programmers

Familiarity with typical computer operation latencies is beneficial. While some numbers may be updated due to technological advancements, they provide a general understanding of performance differences:

| Operation Name                    | Time                     |
|-----------------------------------|--------------------------|
| L1 cache reference                | 0.5 ns                   |
| Branch mispredict                 | 5 ns                     |
| L2 cache reference                | 7 ns                     |
| Mutex lock/unlock                 | 100 ns                   |
| Main memory reference             | 100 ns                   |
| Compress 1KB with Zippy           | 10,000 ns = 10 µs        |
| Send 2KB over 1 Gbps network      | 20,000 ns = 20 µs        |
| Read 1MB sequentially from memory | 250,000 ns = 250 µs      |
| Round trip within same datacenter | 500,000 ns = 500 µs      |
| Disk seek                         | 10,000,000 ns = 10 ms    |
| Read 1MB sequentially from network| 10,000,000 ns = 10 ms    |
| Read 1MB sequentially from disk   | 30,000,000 ns = 30 ms    |
| Send packet CA -> Netherlands -> CA| 150,000,000 ns = 150 ms  |

**Notes:**
*   ns = nanosecond (10^-9 seconds)
*   µs = microsecond (10^-6 seconds = 1,000 ns)
*   ms = millisecond (10^-3 seconds = 1,000 µs = 1,000,000 ns)

Key takeaways from latency analysis:
*   Memory operations are fast; disk operations are slow.
*   Minimize disk seeks.
*   Simple data compression is efficient.
*   Compress data before network transmission when possible.
*   Inter-datacenter communication introduces significant latency.

## Availability Metrics

High availability ensures a system remains operational for a desired period. It is measured as a percentage, with 100% indicating zero downtime. Most services aim for 99% or higher.

A Service Level Agreement (SLA) is a formal commitment between a service provider and a customer regarding uptime. Cloud providers typically set SLAs at 99.9% or above. Uptime is often expressed in "nines," with more nines signifying less downtime. The table below shows the correlation between availability percentage and expected downtime:

| Availability % | Downtime per Day | Downtime per Week | Downtime per Month | Downtime per Year |
|----------------|------------------|-------------------|--------------------|-------------------|
| 99%            | 14.40 minutes    | 1.68 hours        | 7.31 hours         | 3.65 days         |
| 99.99%         | 8.64 seconds     | 1.01 minutes      | 4.38 minutes       | 52.60 minutes     |
| 99.999%        | 864.00 ms        | 6.05 seconds      | 26.30 seconds      | 5.26 minutes      |
| 99.9999%       | 86.40 ms         | 604.80 ms         | 2.63 seconds       | 31.56 seconds     |

## Example: Estimating Twitter's QPS and Storage

(Note: The following figures are for illustrative purposes only and do not represent actual Twitter data.)

**Assumptions:**
*   300 million monthly active users.
*   50% of users are daily active.
*   Users post an average of 2 tweets per day.
*   10% of tweets include media.
*   Data retention: 5 years.

**Estimations:**
*   **Daily Active Users (DAU):** 300 million * 50% = 150 million.
*   **Tweets QPS (Queries Per Second):** (150 million users * 2 tweets/user) / (24 hours/day * 3600 seconds/hour) ≈ 3500 tweets/second.
*   **Peak QPS:** 2 * Tweets QPS = 7000 tweets/second.

**Media Storage Estimation:**
*   **Average Tweet Size Components:**
    *   `tweet_id`: 64 bytes
    *   `text`: 140 bytes
    *   `media`: 1 MB
*   **Daily Media Storage:** 150 million users * 2 tweets/user * 10% (with media) * 1 MB/media = 30 TB/day.
*   **5-Year Media Storage:** 30 TB/day * 365 days/year * 5 years ≈ 55 PB.

## Tips for Back-of-the-Envelope Estimation

Focus on the process, not just the exact result. Interviewers assess your problem-solving approach:

*   **Rounding and Approximation:** Simplify calculations. For example, 99987 / 9.1 can be approximated as 100,000 / 10. Precision is not the primary goal.
*   **Document Assumptions:** Clearly state all assumptions you make during the estimation process.
*   **Label Units:** Always specify units (e.g., "5 MB" instead of "5") to avoid ambiguity.
*   **Practice Common Estimations:** Familiarize yourself with estimating QPS, peak QPS, storage, cache, and server counts. Practice is key. 