---
title: "Design A Chat System"
description: "Explore the design of a scalable chat system supporting one-on-one and group chats, online presence, and multi-device synchronization."
date: "2023-01-13"
order: 13
---

## Design A Chat System

This chapter explores the design of a chat system. Chat applications are ubiquitous, performing diverse functions for different users. It's crucial to pinpoint exact requirements, for instance, whether the focus is on one-on-one or group chat.

### Step 1: Understand the Problem and Establish Design Scope

It is vital to agree on the type of chat app to design. This includes distinguishing between one-on-one chat apps (e.g., Facebook Messenger, WeChat, WhatsApp), office chat apps focusing on group chat (e.g., Slack), or game chat apps (e.g., Discord) that prioritize large group interaction and low voice chat latency.

**Clarification Questions & Assumptions:**
*   **Chat Type:** Supports both one-on-one and group chat.
*   **Platform:** Mobile app and web app.
*   **Scale:** Supports 50 million daily active users (DAU).
*   **Group Member Limit:** Maximum of 100 people per group.
*   **Features:** One-on-one chat, group chat, online indicator. Only text messages are supported initially.
*   **Message Size Limit:** Text length should be less than 100,000 characters.
*   **End-to-End Encryption:** Not required for now, but will be discussed if time allows.
*   **Chat History Storage:** Forever.

**Key Features for this Design (similar to Facebook Messenger):**
*   Low-latency one-on-one chat.
*   Small group chat (max 100 people).
*   Online presence indicator.
*   Multiple device support (same account logged in simultaneously).
*   Push notifications.

We will design a system to support 50 million DAU.

### Step 2: Propose High-Level Design and Get Buy-in

Clients (mobile or web) communicate with a chat service. The chat service is responsible for:

*   Receiving messages from clients.
*   Finding recipients and relaying messages.
*   Holding messages for offline recipients until they come online.

When a client initiates a chat, it connects to the chat service using network protocols. For sending messages, the HTTP protocol with `keep-alive` headers is efficient, as it maintains a persistent connection and reduces TCP handshakes. Many popular chat applications initially used HTTP for sending messages.

The receiver side is more complex, as HTTP is client-initiated. Techniques to simulate server-initiated connections include:

*   **Polling:** Client periodically asks the server for new messages. Can be costly and inefficient due to frequent requests with no new data.
*   **Long Polling:** Client holds a connection open until new messages arrive or a timeout is reached. Upon receiving messages, a new request is immediately sent. Drawbacks include difficulty in maintaining consistent connections across chat servers, no easy way to detect client disconnections, and inefficiency for inactive users.
*   **WebSocket (Preferred):** The most common solution for asynchronous server-to-client updates. It's a bi-directional, persistent connection initiated by the client (upgraded from an HTTP connection). It generally works through firewalls (using ports 80/443). Using WebSocket for both sending and receiving simplifies design and implementation. Efficient connection management is crucial on the server side due to persistent connections.

### High-Level Design Components

While real-time messaging uses WebSocket, other features (signup, login, user profile) can use traditional HTTP request/response. The chat system is divided into three categories:

*   **Stateless Services:** Public-facing request/response services for login, signup, user profiles, etc. They sit behind a load balancer and can be monolithic or microservices. **Service discovery** is a key stateless service, providing clients with chat server DNS hostnames.
*   **Stateful Service (Chat Service):** The chat service is stateful because each client maintains a persistent connection to a specific chat server. Clients usually stick to one server as long as it's available. Service discovery works closely with the chat service to prevent server overload.
*   **Third-Party Integration:** Push notification is the most important integration, informing users of new messages even when the app is not running.

**Scalability Considerations:**
While theoretically a single modern cloud server could handle all user connections for 1M concurrent users (e.g., 10GB RAM needed if 10KB/connection), a single-server design is a major red flag due to being a single point of failure. It's fine to start with this as a conceptual baseline, but acknowledge its limitations for scale.

**Adjusted High-Level Design:**

*   Client maintains a persistent WebSocket connection to a chat server for real-time messaging.
*   Chat servers facilitate message sending/receiving.
*   Presence servers manage online/offline status.
*   API servers handle user login, signup, profile changes, etc.
*   Notification servers send push notifications.
*   A key-value store stores chat history, allowing offline users to retrieve previous chats when they come online.

#### Storage Decisions

Choosing the right database is crucial:

*   **Generic Data (User Profiles, Settings, Friend Lists):** Stored in robust, reliable relational databases. Replication and sharding are used for availability and scalability.
*   **Chat History Data (Unique to Chat Systems):**
    *   Enormous data volume (e.g., Facebook Messenger and WhatsApp process 60 billion messages daily).
    *   Only recent chats are frequently accessed; old chats are rarely looked up.
    *   Supports random access for features like search, mentions, or jumping to specific messages.
    *   Read-to-write ratio for one-on-one chat apps is approximately 1:1.

**Key-value stores are recommended for chat history due to:**
*   Easy horizontal scaling.
*   Very low latency for data access.
*   Relational databases struggle with large indexes for random access (long tail data).
*   Adopted by proven chat applications (e.g., Facebook Messenger uses HBase, Discord uses Cassandra).

#### Data Models (for Key-Value Stores)

*   **Message Table for One-on-One Chat:** Primary key is `message_id` for sequencing. Relying on `created_at` is unreliable as multiple messages can have the same timestamp.

*   **Message Table for Group Chat:** Composite primary key is `(channel_id, message_id)`. `channel_id` serves as the partition key, as all group chat queries operate within a channel.

#### Message ID Generation

`message_id` must be unique and sortable by time (newer IDs are higher). Methods include:
*   **Database `auto_increment`:** Not available in most NoSQL databases.
*   **Global 64-bit sequence number generator (e.g., Snowflake):** Discussed in the "Design a Unique ID Generator in Distributed Systems" chapter.
*   **Local sequence number generator:** IDs are unique only within a group/channel. Easier to implement and sufficient for maintaining message order within a specific conversation.

### Step 3: Design Deep Dive

Common deep dive areas for a chat system include service discovery, messaging flows, and online/offline indicators.

#### Service Discovery

Its main role is to recommend the best chat server to a client based on criteria like geographical location or server capacity. Apache Zookeeper is a popular open-source solution.

**Service Discovery Workflow (Example with Zookeeper):**
1.  User A attempts to log in.
2.  Load balancer sends the login request to API servers.
3.  Backend authenticates the user. Service discovery (Zookeeper) identifies the optimal chat server (e.g., Server 2) and returns its information to User A.
4.  User A establishes a WebSocket connection to Chat Server 2.

#### Message Flows

*   **One-on-One Chat Flow (User A to User B):**
    1.  User A sends message to Chat Server 1.
    2.  Chat Server 1 gets `message_id` from ID generator.
    3.  Chat Server 1 sends message to a message sync queue.
    4.  Message is stored in a key-value store.
    5.a. If User B is online, message is forwarded to Chat Server 2 (where User B is connected).
    5.b. If User B is offline, a push notification is sent from PN servers.
    6.  Chat Server 2 forwards message to User B (via persistent WebSocket).

*   **Message Synchronization Across Multiple Devices:**
    Many users have multiple devices. Each device maintains a `cur_max_message_id` variable, tracking its latest message ID. New messages are those with recipient ID matching the logged-in user and `message_id` greater than `cur_max_message_id`. This allows each device to fetch new messages from the KV store independently.

*   **Small Group Chat Flow:**
    More complex than one-on-one. When User A sends a message in a group (e.g., User A, B, C), the message is copied to each group member's message sync queue (e.g., one for User B, one for User C). This design simplifies message synchronization as each client only checks its own inbox. It's suitable for small groups (e.g., WeChat limits groups to 500 members). For very large groups, copying messages for each member becomes too expensive.

#### Online Presence Indicator

This essential feature (e.g., a green dot next to a username) is managed by presence servers, communicating with clients via WebSocket.

**Online Status Change Triggers:**
*   **User Login:** After WebSocket connection, user's online status and `last_active_at` timestamp are saved in the KV store, and the indicator shows online.
*   **User Logout:** Online status in KV store changes to offline.
*   **User Disconnection:** To avoid frequent status changes from transient network issues, a **heartbeat mechanism** is used. An online client periodically sends heartbeat events to presence servers. If no heartbeat is received within a set time (e.g., x seconds), the user is marked offline.

*   **Online Status Fanout:** Presence servers use a publish-subscribe model. Each friend pair maintains a channel. When User A's status changes, the event is published to relevant channels (e.g., A-B, A-C, A-D), which User B, C, and D subscribe to via WebSocket, receiving real-time updates. This is effective for small groups. For larger groups, constantly informing all members is expensive; instead, online status might be fetched only when a user enters a group or manually refreshes their friend list.

### Step 4: Wrap Up (Additional Considerations)

This chapter presented a chat system architecture supporting one-to-one and small group chat, utilizing WebSocket for real-time communication. Key components included chat servers, presence servers, push notification servers, key-value stores for history, and API servers for other functionalities.

Additional discussion points for an interview could include:

*   **Media File Support:** Extending the chat app for photos/videos. Topics: compression, cloud storage, thumbnails.
*   **End-to-End Encryption:** Discussing how only sender and recipient can read messages (e.g., WhatsApp model).
*   **Client-Side Message Caching:** Reduces data transfer between client and server.
*   **Improved Load Time:** Geographically distributed networks and edge caching (e.g., Slack's Flannel).
*   **Error Handling:**
    *   **Chat Server Errors:** If a chat server goes offline, service discovery (Zookeeper) provides a new server for clients to reconnect.
    *   **Message Resent Mechanism:** Retry and queuing are common techniques for failed message delivery. 