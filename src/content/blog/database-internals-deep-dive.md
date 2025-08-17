---
title: "Database Internals: A Deep Dive into How Distributed Data Systems Work"
description: "Key lessons on storage, transactions, and distributed coordination inspired by Alex Petrov's book."
pubDate: "Aug 17 2025"
tags: ["Databases", "Distributed Systems", "Book Notes"]
---

# Reflections on *Database Internals*

After working through Alex Petrov's *Database Internals*, I came away with appreciation for the machinery that keeps modern data services running. The book is split into two broad parts: the first looks inside single-node storage engines; the second scales out to the world of distributed data systems. Below are the ideas that stuck with me and how they fit together.

### Files, Pages, and Buffers

Most engines organize data in fixed-size pages and use a buffer manager to cache those pages in memory. The buffer pool decides which pages stay hot, which are flushed, and how dirty pages are handled. Write-ahead logging (WAL) ensures durability: changes hit the log before data pages are written back, allowing recovery after crashes.

### B-Tree Family

Traditional relational engines rely on B-tree variants. These balanced trees keep key ranges sorted and support efficient point and range queries. Petrov dives into node layout, sibling pointers, and how rebalancing keeps trees shallow. He also discusses write amplification and techniques like prefix compression that help keep index size in check.

### Log-Structured Merge Trees

As workloads shifted toward heavy writes, the LSM tree emerged. It batches mutations in memory, flushes them as sorted runs on disk, and merges those runs over time. The book highlights compaction strategies, bloom filters to reduce unnecessary reads, and the trade-offs between write throughput and read amplification.

### Concurrency Control

Any multi-user system must juggle concurrent access. Petrov compares locking mechanisms with multiversion concurrency control (MVCC). With MVCC, new versions are written alongside old ones, allowing readers to proceed without blocking writers. Garbage collection, snapshots, and visibility rules are the hidden heroes that keep MVCC performant.

## Distributed Data Building Blocks

Once a single node is mastered, the book expands outward to clusters.

### Replication

Replication is the first step toward fault tolerance. Leader-based replication channels all writes through a primary node and ships a log of changes to followers. This model is simple but can bottleneck. Leaderless replication distributes writes across peers and relies on quorums to establish consistency. The book dissects synchronous vs. asynchronous modes and how lag and failover are handled in both approaches.

### Partitioning

Huge data sets demand partitioning. Horizontal partitioning, or sharding, slices data by key ranges or hash values so no node has to own everything. Petrov explains how consistent hashing minimizes data movement during rebalancing and how systems handle hotspots when certain keys become too popular.

### Membership and Failure Detection

Distributed systems must agree on who is in the cluster. Gossip protocols spread membership information in a scalable way, while failure detectors judge whether a peer is alive based on heartbeats and timeouts. These mechanisms inform replica placement and trigger recovery procedures.

### Consensus Algorithms

To keep replicas in sync, nodes need to agree on the order of operations. The book compares Paxos and Raft, showing how a majority of nodes can safely elect a leader and replicate a log. Petrov spends time on log matching, leader election timeouts, and how configuration changes are applied without splitting the cluster.

### Snapshots and Checkpointing

Periodic snapshots compress history and speed recovery. Instead of replaying an entire log, a system can restore from the latest snapshot and apply only recent mutations. Coordinating snapshots across shards requires barriers to ensure a consistent cut of the distributed state.

### Distributed Transactions

Some workloads need atomic operations across partitions. Two-phase commit (2PC) is the classic approach: a coordinator asks participants to prepare, then commit. Petrov explains why 2PC can block during failures and how algorithms like Percolator and Spanner layer consensus underneath to avoid indefinite locks.

### Secondary Indexes and Query Routing

Distributed indexes introduce new wrinkles. Maintaining secondary indexes across shards may involve scatter-gather queries or co-locating related data. The book walks through covering indexes, global vs. local indexing strategies, and the cost of fan-out reads when indexes live on separate nodes.

## Observability and Operations

Beyond core algorithms, the book touches on practical operations. Monitoring replication lag, tracking disk usage, and alerting on slow compactions are everyday tasks. Petrov advocates for exposing metrics, tracing requests end-to-end, and planning capacity with headroom for spikes.

## Closing Thoughts

By understanding why a B-tree chooses one page over another or how Raft recovers from a network partition, we become better equipped to design and operate data systems of our own.

For anyone building services that depend on reliable storage and coordination, Petrov's guide of the internals is time well spent. The book reminded me that behind every query result lies a dense web of logs, caches, consensus rounds, and carefully orchestrated trade-offs.


