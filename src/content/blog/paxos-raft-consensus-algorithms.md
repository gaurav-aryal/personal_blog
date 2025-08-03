---
title: Paxos and Raft Consensus Algorithms - Building Reliable Distributed Systems
description: A deep dive into consensus algorithms, their applications in distributed systems, and how quorums ensure consistency across unreliable networks.
pubDate: "August 2 2025"
updatedDate: "August 2 2025"
heroImage: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&auto=format&fit=crop&q=60"
tags: ["consensus", "distributed-systems", "paxos", "raft", "algorithms", "system-design"]
---

# Paxos and Raft Consensus Algorithms - Building Reliable Distributed Systems

Consensus algorithms are the backbone of reliable distributed systems. They enable multiple nodes to agree on a single value or sequence of values, even when some nodes fail or network partitions occur. In this comprehensive guide, we'll explore two of the most important consensus algorithms: **Paxos** and **Raft**, along with the concept of **quorums** that make them work.

## The Consensus Problem

Before diving into specific algorithms, let's understand the consensus problem:

**Given a set of nodes that can fail or become unreachable, how do we ensure all nodes agree on a single value?**

This seemingly simple problem becomes incredibly complex when you consider:
- **Network partitions**: Nodes can't communicate with each other
- **Node failures**: Some nodes may crash or become unresponsive
- **Byzantine failures**: Nodes may behave maliciously (though Paxos/Raft don't handle this)
- **Timing issues**: Messages may be delayed or lost

## What is a Quorum?

A **quorum** is a subset of nodes whose agreement is sufficient to make a decision. The key insight is that if two quorums have any overlap, they cannot make conflicting decisions.

### Quorum Properties

1. **Intersection Property**: Any two quorums must have at least one node in common
2. **Majority Quorum**: Most common approach where a quorum is any majority of nodes
3. **Fault Tolerance**: System can tolerate up to `⌊(n-1)/2⌋` failures in an n-node system

### Quorum Examples

```java
public class QuorumExample {
    private final int totalNodes;
    private final int quorumSize;
    
    public QuorumExample(int totalNodes) {
        this.totalNodes = totalNodes;
        this.quorumSize = (totalNodes / 2) + 1; // Majority
    }
    
    public boolean isQuorum(int respondingNodes) {
        return respondingNodes >= quorumSize;
    }
    
    public int getMaxFailures() {
        return totalNodes - quorumSize;
    }
}

// Example usage
QuorumExample quorum = new QuorumExample(5); // 5 nodes
System.out.println("Quorum size: " + quorum.quorumSize); // 3
System.out.println("Max failures: " + quorum.getMaxFailures()); // 2
```

## Paxos Algorithm

Paxos, developed by Leslie Lamport in 1989, is one of the most influential consensus algorithms. It's used in systems like Google's Chubby and Apache ZooKeeper.

### Paxos Phases

Paxos operates in two phases:

#### Phase 1: Prepare Phase
1. **Proposer** sends prepare request with proposal number `n`
2. **Acceptors** respond with:
   - Promise not to accept proposals numbered less than `n`
   - Highest numbered proposal they've accepted (if any)

#### Phase 2: Accept Phase
1. **Proposer** sends accept request with value `v`
2. **Acceptors** accept the proposal if they haven't promised a higher number

### Paxos Implementation

```java
public class PaxosNode {
    private int nodeId;
    private int proposalNumber;
    private Object acceptedValue;
    private int acceptedProposalNumber;
    private Set<Integer> promises;
    
    public PaxosNode(int nodeId) {
        this.nodeId = nodeId;
        this.proposalNumber = 0;
        this.promises = new HashSet<>();
    }
    
    public PrepareResponse prepare(int proposalNum) {
        if (proposalNum > proposalNumber) {
            proposalNumber = proposalNum;
            promises.add(proposalNum);
            
            return new PrepareResponse(
                true, // promise
                acceptedValue,
                acceptedProposalNumber
            );
        }
        return new PrepareResponse(false, null, 0);
    }
    
    public boolean accept(int proposalNum, Object value) {
        if (promises.contains(proposalNum)) {
            acceptedValue = value;
            acceptedProposalNumber = proposalNum;
            return true;
        }
        return false;
    }
}

class PrepareResponse {
    boolean promised;
    Object acceptedValue;
    int acceptedProposalNumber;
    
    public PrepareResponse(boolean promised, Object value, int proposalNum) {
        this.promised = promised;
        this.acceptedValue = value;
        this.acceptedProposalNumber = proposalNum;
    }
}
```

### Paxos Proposer Implementation

```java
public class PaxosProposer {
    private int proposalNumber;
    private List<PaxosNode> acceptors;
    
    public Object propose(Object value) {
        proposalNumber++;
        
        // Phase 1: Prepare
        int promises = 0;
        Object highestAcceptedValue = null;
        int highestAcceptedNumber = 0;
        
        for (PaxosNode acceptor : acceptors) {
            PrepareResponse response = acceptor.prepare(proposalNumber);
            if (response.promised) {
                promises++;
                if (response.acceptedProposalNumber > highestAcceptedNumber) {
                    highestAcceptedNumber = response.acceptedProposalNumber;
                    highestAcceptedValue = response.acceptedValue;
                }
            }
        }
        
        // Check if we have a quorum
        if (promises < (acceptors.size() / 2) + 1) {
            return null; // No consensus
        }
        
        // Phase 2: Accept
        Object valueToPropose = (highestAcceptedValue != null) ? 
            highestAcceptedValue : value;
        
        int accepts = 0;
        for (PaxosNode acceptor : acceptors) {
            if (acceptor.accept(proposalNumber, valueToPropose)) {
                accepts++;
            }
        }
        
        if (accepts >= (acceptors.size() / 2) + 1) {
            return valueToPropose; // Consensus reached
        }
        
        return null; // No consensus
    }
}
```

## Raft Algorithm

Raft, developed by Diego Ongaro and John Ousterhout in 2014, was designed to be more understandable than Paxos while providing the same safety guarantees.

### Raft Key Concepts

1. **Leader Election**: One node becomes leader, others are followers
2. **Log Replication**: Leader replicates log entries to followers
3. **Safety**: Only committed entries are applied to state machine

### Raft Node States

```java
public enum RaftState {
    FOLLOWER,
    CANDIDATE,
    LEADER
}

public class RaftNode {
    private int nodeId;
    private RaftState state;
    private int currentTerm;
    private int votedFor;
    private int leaderId;
    
    // Log entries
    private List<LogEntry> log;
    private int commitIndex;
    private int lastApplied;
    
    // Leader state
    private Map<Integer, Integer> nextIndex;
    private Map<Integer, Integer> matchIndex;
    
    public RaftNode(int nodeId) {
        this.nodeId = nodeId;
        this.state = RaftState.FOLLOWER;
        this.currentTerm = 0;
        this.votedFor = -1;
        this.log = new ArrayList<>();
        this.commitIndex = 0;
        this.lastApplied = 0;
    }
}
```

### Raft Leader Election

```java
public class RaftLeaderElection {
    private RaftNode node;
    private Random random;
    private int electionTimeout;
    
    public void startElection() {
        node.setState(RaftState.CANDIDATE);
        node.setCurrentTerm(node.getCurrentTerm() + 1);
        node.setVotedFor(node.getNodeId());
        
        // Request votes from all other nodes
        int votes = 1; // Vote for self
        
        for (RaftNode otherNode : getAllNodes()) {
            if (otherNode.getNodeId() != node.getNodeId()) {
                VoteRequest request = new VoteRequest(
                    node.getCurrentTerm(),
                    node.getNodeId(),
                    node.getLog().size() - 1,
                    node.getLog().get(node.getLog().size() - 1).getTerm()
                );
                
                VoteResponse response = otherNode.requestVote(request);
                if (response.isVoteGranted()) {
                    votes++;
                }
            }
        }
        
        // Check if we have a quorum
        if (votes >= (getAllNodes().size() / 2) + 1) {
            becomeLeader();
        } else {
            // Election timeout, try again
            scheduleElectionTimeout();
        }
    }
    
    private void becomeLeader() {
        node.setState(RaftState.LEADER);
        node.setLeaderId(node.getNodeId());
        
        // Initialize leader state
        for (RaftNode otherNode : getAllNodes()) {
            if (otherNode.getNodeId() != node.getNodeId()) {
                node.getNextIndex().put(otherNode.getNodeId(), node.getLog().size());
                node.getMatchIndex().put(otherNode.getNodeId(), 0);
            }
        }
        
        // Start sending heartbeats
        startHeartbeat();
    }
}
```

### Raft Log Replication

```java
public class RaftLogReplication {
    private RaftNode leader;
    
    public void replicateLog(Object command) {
        // Add entry to leader's log
        LogEntry entry = new LogEntry(
            leader.getCurrentTerm(),
            command
        );
        leader.getLog().add(entry);
        
        // Send AppendEntries RPC to all followers
        for (RaftNode follower : getFollowers()) {
            sendAppendEntries(follower, entry);
        }
    }
    
    private void sendAppendEntries(RaftNode follower, LogEntry entry) {
        AppendEntriesRequest request = new AppendEntriesRequest(
            leader.getCurrentTerm(),
            leader.getNodeId(),
            leader.getLog().size() - 2, // prevLogIndex
            leader.getLog().get(leader.getLog().size() - 2).getTerm(), // prevLogTerm
            entry,
            leader.getCommitIndex()
        );
        
        AppendEntriesResponse response = follower.appendEntries(request);
        
        if (response.isSuccess()) {
            // Update match index
            leader.getMatchIndex().put(follower.getNodeId(), 
                leader.getLog().size() - 1);
            leader.getNextIndex().put(follower.getNodeId(), 
                leader.getLog().size());
            
            // Try to commit
            tryCommit();
        } else {
            // Decrement nextIndex and retry
            int nextIndex = leader.getNextIndex().get(follower.getNodeId());
            leader.getNextIndex().put(follower.getNodeId(), nextIndex - 1);
        }
    }
    
    private void tryCommit() {
        // Find the highest index that can be committed
        for (int i = leader.getLog().size() - 1; i > leader.getCommitIndex(); i--) {
            int count = 1; // Leader has this entry
            
            for (RaftNode follower : getFollowers()) {
                if (leader.getMatchIndex().get(follower.getNodeId()) >= i) {
                    count++;
                }
            }
            
            // Check if majority has this entry
            if (count >= (getAllNodes().size() / 2) + 1) {
                leader.setCommitIndex(i);
                break;
            }
        }
    }
}
```

## Applications of Consensus Algorithms

### 1. Distributed Databases

**Apache Cassandra**
- Uses a variant of Paxos for lightweight transactions
- Ensures consistency across multiple data centers
- Handles network partitions gracefully

```java
// Cassandra Paxos implementation example
public class CassandraPaxos {
    public boolean executeLightweightTransaction(String key, Object value) {
        // Phase 1: Prepare
        List<PrepareResponse> responses = prepare(key);
        
        // Check quorum
        if (!hasQuorum(responses)) {
            return false;
        }
        
        // Phase 2: Propose
        List<ProposeResponse> proposeResponses = propose(key, value);
        
        // Phase 3: Commit
        return commit(key, value);
    }
}
```

**MongoDB**
- Uses Raft for replica set elections
- Ensures data consistency across replicas
- Automatic failover when primary fails

### 2. Distributed Key-Value Stores

**etcd**
- Built on Raft consensus
- Used by Kubernetes for configuration storage
- Provides strong consistency guarantees

```java
// etcd-like key-value store with Raft
public class EtcdStore {
    private RaftNode raftNode;
    private Map<String, String> keyValueStore;
    
    public void put(String key, String value) {
        // Create log entry
        LogEntry entry = new LogEntry(
            raftNode.getCurrentTerm(),
            new PutCommand(key, value)
        );
        
        // Replicate through Raft
        if (raftNode.isLeader()) {
            replicateLog(entry);
        }
    }
    
    public String get(String key) {
        // Read from local state
        return keyValueStore.get(key);
    }
}
```

### 3. Service Discovery and Configuration

**Apache ZooKeeper**
- Uses Zab (ZooKeeper Atomic Broadcast) protocol
- Provides coordination services for distributed applications
- Used by Kafka, Hadoop, and many other systems

```java
// ZooKeeper-like coordination service
public class ZooKeeperService {
    private RaftNode consensusNode;
    private TreeMap<String, NodeData> dataTree;
    
    public void createNode(String path, String data) {
        CreateNodeCommand command = new CreateNodeCommand(path, data);
        LogEntry entry = new LogEntry(
            consensusNode.getCurrentTerm(),
            command
        );
        
        replicateLog(entry);
    }
    
    public String getNodeData(String path) {
        return dataTree.get(path).getData();
    }
}
```

### 4. Message Queues

**Apache Kafka**
- Uses ZooKeeper for controller election
- Ensures message ordering and durability
- Handles partition leadership changes

### 5. Blockchain Systems

**Hyperledger Fabric**
- Uses PBFT (Practical Byzantine Fault Tolerance)
- Ensures consensus among network participants
- Handles malicious nodes

## Why These Applications Use Consensus

### 1. **Data Consistency**
- Ensures all nodes see the same data
- Prevents split-brain scenarios
- Maintains ACID properties

### 2. **Fault Tolerance**
- System continues operating despite node failures
- Automatic recovery from failures
- No single point of failure

### 3. **High Availability**
- System remains available during failures
- Automatic failover mechanisms
- Load distribution across nodes

### 4. **Linearizability**
- Operations appear to execute atomically
- Strong consistency guarantees
- Predictable behavior

## Quorum in Practice

### Read Quorums vs Write Quorums

```java
public class QuorumSystem {
    private int totalNodes;
    private int readQuorum;
    private int writeQuorum;
    
    public QuorumSystem(int totalNodes) {
        this.totalNodes = totalNodes;
        // Common pattern: read quorum + write quorum > total nodes
        this.readQuorum = (totalNodes / 2) + 1;
        this.writeQuorum = (totalNodes / 2) + 1;
    }
    
    public boolean read(Object key) {
        int responses = 0;
        Object value = null;
        
        for (Node node : getAllNodes()) {
            Object nodeValue = node.read(key);
            if (nodeValue != null) {
                responses++;
                if (value == null) {
                    value = nodeValue;
                }
            }
        }
        
        return responses >= readQuorum;
    }
    
    public boolean write(Object key, Object value) {
        int responses = 0;
        
        for (Node node : getAllNodes()) {
            if (node.write(key, value)) {
                responses++;
            }
        }
        
        return responses >= writeQuorum;
    }
}
```

### Quorum Types

1. **Majority Quorum**: Most common, requires >50% of nodes
2. **Weighted Quorum**: Nodes have different weights
3. **Grid Quorum**: Nodes arranged in a grid pattern
4. **Hierarchical Quorum**: Nodes organized in a hierarchy

## Performance Considerations

### Latency vs Consistency

```java
public class ConsistencyLevel {
    public enum Level {
        ONE,           // Single node response
        QUORUM,        // Majority response
        ALL            // All nodes response
    }
    
    public Object readWithConsistency(Object key, Level level) {
        switch (level) {
            case ONE:
                return readFromAnyNode(key);
            case QUORUM:
                return readWithQuorum(key);
            case ALL:
                return readFromAllNodes(key);
            default:
                throw new IllegalArgumentException();
        }
    }
}
```

### Network Partition Handling

```java
public class NetworkPartitionHandler {
    public void handlePartition(List<Node> availableNodes) {
        // Check if we have a quorum
        if (availableNodes.size() >= getQuorumSize()) {
            // Continue normal operation
            continueOperation();
        } else {
            // Stop accepting writes, only allow reads
            enterReadOnlyMode();
        }
    }
    
    private void enterReadOnlyMode() {
        // Reject write operations
        // Allow read operations from available nodes
        // Wait for network partition to resolve
    }
}
```

## Best Practices

### 1. **Choose the Right Algorithm**
- **Paxos**: When you need proven correctness
- **Raft**: When you need understandability
- **PBFT**: When you need Byzantine fault tolerance

### 2. **Configure Appropriate Timeouts**
```java
public class TimeoutConfig {
    private int electionTimeout = 1000; // ms
    private int heartbeatInterval = 100; // ms
    private int requestTimeout = 5000; // ms
    
    public void setTimeouts(int electionTimeout, int heartbeatInterval, int requestTimeout) {
        this.electionTimeout = electionTimeout;
        this.heartbeatInterval = heartbeatInterval;
        this.requestTimeout = requestTimeout;
    }
}
```

### 3. **Monitor Consensus Health**
```java
public class ConsensusMonitor {
    public void monitorHealth() {
        // Check leader election frequency
        // Monitor log replication lag
        // Track commit rates
        // Alert on quorum violations
    }
}
```

## Conclusion

Consensus algorithms like Paxos and Raft are fundamental to building reliable distributed systems. They provide the foundation for:

- **Data consistency** across multiple nodes
- **Fault tolerance** in the face of failures
- **High availability** through automatic recovery
- **Strong guarantees** for critical applications

Understanding quorums and how they ensure consistency is crucial for designing robust distributed systems. Whether you're building a database, message queue, or blockchain, consensus algorithms provide the reliability guarantees your system needs.

The choice between Paxos and Raft often depends on your specific requirements:
- **Paxos**: More complex but battle-tested
- **Raft**: Easier to understand and implement

Both algorithms provide the same safety guarantees, but Raft's simplicity makes it more accessible for new implementations.

Remember: Consensus is not just about agreeing on values—it's about building systems that can withstand the chaos of distributed computing while maintaining consistency and availability. 