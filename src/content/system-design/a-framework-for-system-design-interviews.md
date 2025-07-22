---
title: "A Framework For System Design Interviews"
description: "A simple and effective framework to solve system design interview problems."
date: "2023-01-04"
order: 4
---

## A 4-Step Process for Effective System Design Interviews

Every system design interview is unique, and there's no one-size-fits-all solution. However, a structured approach can help you navigate common ground in any system design interview.

### Step 1: Understand the Problem and Establish Design Scope

Avoid rushing to a solution without fully understanding the requirements. This interview is not a trivia contest; there's no single "right" answer. Take your time to think deeply and ask clarifying questions. As an engineer, it's crucial to ask the right questions, make proper assumptions, and gather all necessary information to build a system.

When you ask a question, the interviewer will either provide a direct answer or ask you to make assumptions. If you make assumptions, document them, as they might be relevant later.

**Types of questions to ask:**
*   What specific features are we building?
*   How many users does the product have?
*   What are the anticipated scaling requirements in 3, 6, and 12 months?
*   What is the company's technology stack? Can existing services be leveraged?

**Example: Designing a News Feed System**

*   **Candidate:** Is this for a mobile app, web app, or both?
*   **Interviewer:** Both.
*   **Candidate:** What are the most important features?
*   **Interviewer:** Ability to make a post and see friends' news feeds.
*   **Candidate:** Is the news feed sorted chronologically or by a specific order (e.g., weighted by friend closeness)?
*   **Interviewer:** To keep it simple, assume reverse chronological order.
*   **Candidate:** How many friends can a user have?
*   **Interviewer:** 5000.
*   **Candidate:** What is the traffic volume?
*   **Interviewer:** 10 million daily active users (DAU).
*   **Candidate:** Can the feed contain images, videos, or just text?
*   **Interviewer:** It can contain media files, including both images and videos.

These sample questions demonstrate how to clarify requirements and ambiguities.

### Step 2: Propose High-Level Design and Get Buy-in

Collaborate with your interviewer to develop a high-level design and seek their agreement. Think of your interviewer as a teammate.

*   **Initial Blueprint:** Sketch out key components (clients, APIs, web servers, data stores, cache, CDN, message queues, etc.) using box diagrams.
*   **Back-of-the-Envelope Calculations:** Evaluate if your design can handle the scale constraints. Discuss these calculations with your interviewer.
*   **Concrete Use Cases:** Walk through a few concrete use cases to frame the design and uncover edge cases.
*   **API Endpoints and Database Schema:** Discuss whether to include these details based on the problem's scope. For large problems like Google Search, it might be too low-level; for smaller problems, it could be appropriate. Always communicate with your interviewer.

**Example: Designing a News Feed System (High-Level)**

At a high level, the design can be divided into two flows:

*   **Feed Publishing:** When a user publishes a post, data is written to a cache/database, and the post is propagated to friends' news feeds.
*   **News Feed Building:** The news feed is constructed by aggregating friends' posts in reverse chronological order.

### Step 3: Design Deep Dive

At this stage, you should have:
*   Agreed on overall goals and feature scope.
*   Sketched a high-level design blueprint.
*   Received feedback on the high-level design.
*   Initial ideas about areas for a deep dive based on feedback.

Work with the interviewer to prioritize architectural components for deeper discussion. The focus can vary: sometimes it's high-level, other times it's specific component performance or bottlenecks. For instance, in a URL shortener design, the hash function might be a deep dive area. For a chat system, latency reduction and online/offline status are relevant topics.

**Time management is crucial.** Avoid getting lost in minor details that don't showcase your core abilities. Focus on demonstrating your design skills for scalable systems.

**Example: News Feed System Deep Dive**

After agreeing on the high-level design for a news feed system, you might delve into two critical use cases:

1.  Feed publishing.
2.  News feed retrieval.

### Step 4: Wrap Up

In this final step, the interviewer may ask follow-up questions or allow you to discuss additional points. Use this opportunity to demonstrate critical thinking.

*   **Identify Bottlenecks and Improvements:** Never claim your design is perfect. There's always room for improvement. Discuss potential bottlenecks and how to address them.
*   **Recap Your Design:** Briefly summarize your proposed solution, especially if you discussed multiple approaches.
*   **Error Handling:** Discuss how to handle server failures, network loss, and other exceptions gracefully.
*   **Operational Issues:** Mention monitoring metrics, error logs, and deployment strategies.
*   **Scaling for the Future:** Discuss how the current design would evolve to support larger scales (e.g., from 1 million to 10 million users).
*   **Future Refinements:** Propose additional improvements if you had more time.

**Dos and Don'ts:**

**Dos:**
*   Always ask for clarification and don't assume.
*   Thoroughly understand the problem requirements.
*   Recognize that there's no single "right" or "best" answer; solutions depend on requirements.
*   Communicate your thought process openly.
*   Suggest multiple approaches when appropriate.
*   After agreeing on a blueprint, deep dive into critical components.
*   Collaborate with your interviewer.
*   Never give up.

**Don'ts:**
*   Be unprepared for typical interview questions.
*   Jump into a solution without clarifying requirements.
*   Go into excessive detail on one component initially; provide a high-level overview first.
*   Hesitate to ask for hints if stuck.
*   Think in silence; communicate constantly.
*   Assume the interview is over after presenting your design. Seek feedback early and often.

## Time Allocation (Rough Guide for a 45-Minute Interview):

*   **Step 1 (Understand the problem and establish design scope):** 3 - 10 minutes
*   **Step 2 (Propose high-level design and get buy-in):** 10 - 15 minutes
*   **Step 3 (Design deep dive):** 10 - 25 minutes
*   **Step 4 (Wrap up):** 3 - 5 minutes 