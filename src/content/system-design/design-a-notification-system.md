---
title: "Design A Notification System"
description: "Design a scalable notification system supporting mobile push, SMS, and email, addressing reliability, delivery, and user preferences."
date: "2023-01-11"
order: 11
---

## Design A Notification System

A notification system is a crucial feature in many applications, alerting users to important information like news, updates, events, and offers. This chapter focuses on designing a scalable notification system.

Notifications come in various formats, including mobile push notifications, SMS messages, and emails.

### Step 1: Understand the Problem and Establish Design Scope

Building a system capable of sending millions of notifications daily requires a deep understanding of the notification ecosystem. The interview question is intentionally open-ended, so clarifying requirements is key.

**Clarification Questions & Assumptions:**
*   **Supported Notification Types:** Push notification (iOS, Android), SMS messages, and email.
*   **Real-time System:** Soft real-time; notifications should be delivered as soon as possible, with slight delays acceptable under high workload.
*   **Supported Devices:** iOS devices, Android devices, and laptop/desktop.
*   **Notification Triggers:** Client applications or server-side schedules.
*   **User Opt-out:** Yes, users can opt-out of receiving notifications.
*   **Daily Notification Volume:** 10 million mobile push, 1 million SMS, and 5 million emails.

### Step 2: Propose High-Level Design and Get Buy-in

This section outlines the high-level design for supporting various notification types, covering contact information gathering and the notification sending/receiving flow.

#### Different Types of Notifications

*   **iOS Push Notification:** Involves a **Provider** (your service) sending notification requests with a unique **Device Token** and **Payload** to **Apple Push Notification Service (APNS)**, which then propagates them to **iOS Devices**.

*   **Android Push Notification:** Similar to iOS, but uses **Firebase Cloud Messaging (FCM)** instead of APNS to deliver notifications to **Android Devices**.

*   **SMS Message:** Typically uses third-party SMS services like Twilio or Nexmo for delivery.

*   **Email:** Many companies opt for commercial email services such as SendGrid or Mailchimp for better delivery rates and analytics, rather than setting up their own email servers.

Third-party services are integral to the design, handling the actual delivery to user devices.

#### Contact Information Gathering Flow

To send notifications, the system needs to collect mobile device tokens, phone numbers, or email addresses. When a user installs the app or signs up, API servers collect this contact information and store it in a database. Email addresses and phone numbers are typically stored in a `user` table, while device tokens (a user can have multiple devices) are stored in a `device` table, allowing push notifications to all user devices.

#### Notification Sending/Receiving Flow (Initial Design & Improvements)

**Initial High-Level Design:**

*   **Services 1 to N:** Microservices, cron jobs, or distributed systems that trigger notification events (e.g., a billing service for payment reminders).
*   **Notification System:** A central server providing APIs for services and building notification payloads for third-party services.
*   **Third-Party Services:** Responsible for delivering notifications. Extensibility is key, as services might be unavailable in new markets (e.g., FCM in China) or in the future.
*   **iOS, Android, SMS, Email:** User devices receiving notifications.

**Problems Identified in Initial Design:**
*   **Single Point of Failure (SPOF):** A single notification server is a SPOF.
*   **Hard to Scale:** Challenges in independently scaling databases, caches, and processing components.
*   **Performance Bottleneck:** Resource-intensive processing (HTML rendering, third-party responses) can overload a single system during peak hours.

**Improved High-Level Design:**

To address these challenges, the improved design incorporates:
*   Moving the database and cache out of the notification server.
*   Adding more notification servers with automatic horizontal scaling.
*   Introducing message queues to decouple system components.

**Components in Improved Design:**

*   **Services 1 to N:** Various services that send notifications via APIs provided by notification servers.
*   **Notification Servers:** Provide internal or authenticated APIs for services, perform basic validations, fetch data for rendering (from cache/DB), and put notification data into message queues for parallel processing.
*   **Cache:** Stores frequently accessed user info, device info, and notification templates.
*   **DB:** Stores user data, notification logs, settings, etc.
*   **Message Queues:** Decouple components and act as buffers for high volumes. Each notification type has a distinct queue to isolate outages.
*   **Workers:** Servers that pull notification events from message queues and send them to the respective third-party services.
*   **Third-Party Services:** (As explained above).
*   **iOS, Android, SMS, Email:** (As explained above).

**Notification Sending Workflow:**
1.  A service calls Notification Server APIs.
2.  Notification Servers fetch metadata from cache/database.
3.  A notification event is sent to the corresponding message queue (e.g., iOS PN queue).
4.  Workers pull events from message queues.
5.  Workers send notifications to third-party services.
6.  Third-party services deliver notifications to user devices.

### Step 3: Design Deep Dive

We'll now explore reliability and additional considerations, followed by an updated system design.

#### Reliability

*   **Prevent Data Loss:** Notifications cannot be lost, though delays or re-ordering are acceptable. This is achieved by persisting notification data in a database (notification log) and implementing a retry mechanism.
*   **Exactly-Once Delivery:** While generally delivered exactly once, distributed systems can sometimes result in duplicates. A deduplication mechanism (checking event ID upon arrival) can reduce this. Achieving true exactly-once delivery in distributed systems is complex.

#### Additional Components and Considerations

A comprehensive notification system involves more than just basic sending and receiving:

*   **Notification Template:** Reusable, preformatted notifications allowing customization of parameters, styling, and tracking links. Benefits include consistent format, reduced errors, and time-saving.
*   **Notification Setting:** Users can control their notification preferences (e.g., opt-in/out per channel). This information is stored (e.g., `user_id`, `channel`, `opt_in`) and checked before sending any notification.
*   **Rate Limiting:** Limits the number of notifications a user receives to prevent overwhelming them, which could lead to users disabling notifications entirely.
*   **Retry Mechanism:** If a third-party service fails to send a notification, it's re-queued for retry. Persistent failures trigger alerts to developers.
*   **Security in Push Notifications:** For mobile apps, `appKey` and `appSecret` are used to secure push notification APIs, ensuring only authenticated clients can send notifications.
*   **Monitor Queued Notifications:** A key metric. A large number indicates workers are not processing events fast enough, requiring more workers to avoid delivery delays.
*   **Event Tracking:** Analytics services track notification metrics (open rate, click rate, engagement) to understand customer behavior. Integration with an analytics service is essential.

#### Updated Design

Integrating these components, the updated notification system design includes:

*   **Notification Servers:** Now incorporate authentication and rate-limiting.
*   **Retry Mechanism:** Added to handle failures; notifications are re-queued and retried a predefined number of times.
*   **Notification Templates:** Provide efficient and consistent notification creation.
*   **Monitoring and Tracking Systems:** For system health checks and future improvements.

### Step 4: Wrap Up

Notifications are vital for keeping users informed. This chapter outlined the design of a scalable notification system supporting push, SMS, and email, leveraging message queues for decoupling.

Key areas explored in depth included:
*   **Reliability:** Robust retry mechanisms to minimize failures.
*   **Security:** Using `appKey`/`appSecret` for authenticated notification sending.
*   **Tracking and Monitoring:** Capturing important statistics throughout the notification flow.
*   **User Settings:** Respecting user preferences (opt-out).
*   **Rate Limiting:** Implementing frequency capping to enhance user experience.

Designing a comprehensive notification system involves balancing functionality with reliability, security, and user experience. 