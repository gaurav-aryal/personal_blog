---
title: Design A Unique ID Generator In Distributed Systems
description: Explore various approaches to designing a unique ID generator for distributed systems, focusing on scalability, uniqueness, and ordering.
date: "June 1 2025"
order: 8
---

## Requirements and Scope

When designing a distributed ID generator, it's crucial to define key requirements:

*   **Uniqueness**: Each generated ID must be globally unique.
*   **Numerical**: IDs should consist only of numerical values.
*   **Size**: IDs should fit within a 64-bit integer.
*   **Order**: IDs should be roughly ordered by creation time (e.g., IDs generated later in the day are larger than those from earlier).
*   **Scale**: The system must support high throughput, e.g., generating over 10,000 IDs per second.

## High-Level Design Approaches

Several strategies exist for distributed unique ID generation, each with its own advantages and drawbacks:

### Multi-Master Replication

This approach leverages database `auto_increment` by configuring multiple master databases. To avoid conflicts, each master is assigned an increment greater than 1, based on the number of masters. For example, with `k` masters, each server increments its ID by `k`.

**Pros:**
*   Leverages existing database features.

**Cons:**
*   Difficult to scale across multiple data centers.
*   IDs are not globally time-ordered across servers.
*   Poor scalability when servers are added or removed dynamically.

### Universally Unique Identifier (UUID)

UUIDs are 128-bit numbers designed for global uniqueness. They can be generated independently by any system without central coordination.

**Pros:**
*   Simple generation without server coordination.
*   Highly scalable as each server generates its own IDs.
*   Extremely low probability of collisions.

**Cons:**
*   128-bit length may exceed the 64-bit requirement.
*   IDs are not inherently time-ordered.
*   Can contain non-numeric characters.

### Ticket Server

Inspired by solutions like Flickr's ticket servers, this method centralizes ID generation to a single database server (the "Ticket Server") that uses `auto_increment` to dispense blocks of IDs. Clients request a block of IDs from the ticket server and then generate unique IDs within that block locally.

**Pros:**
*   Generates numeric, sequential IDs.
*   Simple to implement for small to medium scale.

**Cons:**
*   Single point of failure. Redundancy introduces complexities like data synchronization.

### Snowflake-like Approach

Inspired by Twitter's Snowflake system, this approach constructs a 64-bit ID by combining different components. This method is highly flexible and can meet all our specified requirements.

```
64-bit ID Structure:

Sign bit (1 bit) | Timestamp (41 bits) | Datacenter ID (5 bits) | Machine ID (5 bits) | Sequence Number (12 bits)

Total: 1 + 41 + 5 + 5 + 12 = 64 bits
```

**Component Breakdown:**

*   **Sign bit (1 bit):** Always 0, reserved for future use or distinguishing signed/unsigned numbers.
*   **Timestamp (41 bits):** Represents milliseconds since a custom epoch (e.g., Twitter's epoch: Nov 04, 2010, 01:42:54 UTC). This ensures IDs are time-ordered.
    *   41 bits provide enough range for approximately 69 years of unique timestamps.
*   **Datacenter ID (5 bits):** Allows for 2^5 = 32 distinct data centers.
*   **Machine ID (5 bits):** Allows for 2^5 = 32 machines per data center.
*   **Sequence Number (12 bits):** A counter that increments for each ID generated within the same millisecond on a specific machine. It resets to 0 every millisecond.
    *   12 bits support 2^12 = 4096 unique IDs per millisecond per machine.

## Detailed Design: Snowflake-like ID Generator

Datacenter IDs and machine IDs are typically configured at system startup and remain static. Timestamp and sequence numbers are dynamic during operation.

**Timestamp Generation:** The 41-bit timestamp ensures chronological ordering. This field measures milliseconds since a chosen epoch. Using a recent epoch maximizes the lifespan of the ID generator before the timestamp overflows (approx. 69 years).

**Sequence Number Management:** The 12-bit sequence number allows for rapid ID generation within a single millisecond on a given machine. If multiple IDs are requested within the same millisecond, this counter increments. If the counter reaches its maximum (4095) within a millisecond, the system must wait for the next millisecond to reset the counter.

## Considerations and Extensions

*   **Clock Synchronization:** This design assumes synchronized clocks across ID generation servers. In practice, Network Time Protocol (NTP) or similar solutions are crucial for maintaining consistent timestamps across distributed machines.
*   **Section Length Tuning:** The bit allocation for each section can be adjusted based on specific needs. For example, if concurrency is low but a longer lifespan is desired, more bits can be allocated to the timestamp and fewer to the sequence number.
*   **High Availability:** As a critical service, the ID generator itself should be highly available, often achieved through redundant deployments and failure detection mechanisms.

## Summary

The Snowflake-like approach provides a robust and scalable solution for generating unique, time-ordered, numerical 64-bit IDs in distributed systems. Its segmented structure offers flexibility and addresses the limitations of simpler methods like multi-master replication or UUIDs, fulfilling the requirements for high-throughput, distributed ID generation.

## Reference Materials
*   [1] Universally unique identifier: https://en.wikipedia.org/wiki/Universally_unique_identifier
*   [2] Ticket Servers: Distributed Unique Primary Keys on the Cheap: https://code.flickr.net/2010/02/08/ticket-servers-distributed-unique-primary-keys-on-the-cheap/
*   [3] Announcing Snowflake: https://blog.twitter.com/engineering/en_us/a/2010/announcing-snowflake.html
*   [4] Network time protocol: https://en.wikipedia.org/wiki/Network_Time_Protocol 