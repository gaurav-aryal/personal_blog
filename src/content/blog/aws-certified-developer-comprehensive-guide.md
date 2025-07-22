---
title: "Mastering AWS"
date: 2025-06-16
pubDate: "2025-06-16"
draft: false
description: "A deep dive into AWS core services and architecture concepts based on Stéphane Maarek's AWS Developer material."
tags: ["AWS", "Cloud Computing", "Serverless", "IAM", "DevOps", "Storage"]
---

# Mastering AWS

This guide presents a structured, service-by-service breakdown of key AWS concepts as outlined by Stéphane Maarek in his AWS Developer course material. Whether you're building scalable APIs, managing access policies, or architecting distributed systems, this guide summarizes what matters most.

## Cloud Fundamentals & Global Infrastructure

### What Is AWS?
AWS (Amazon Web Services) is the leading cloud platform offering over 200 services. Its strengths lie in:

- Elasticity & On-Demand Provisioning
- Global Reach: Dozens of Regions, each with multiple Availability Zones (AZs)
- Edge Locations: Hundreds of content delivery nodes around the globe

### Key Infrastructure Components:
- Regions: Isolated geographical areas (e.g., us-east-1, eu-west-3)
- Availability Zones: Physically separated data centers within a region
- Points of Presence: Used by CloudFront for content delivery

## Identity & Access Management (IAM)

IAM is AWS's global security system for managing authentication and authorization.

### Key Concepts:
- Users & Groups: Create individual users and assign them to groups for easier permission management
- Policies: JSON documents defining what actions are allowed or denied on which resources
- Roles: Assign temporary permissions to AWS services (e.g., Lambda, EC2)
- Access Keys: Used for programmatic access via CLI or SDK
- Multi-Factor Authentication (MFA): Essential for securing the root account and IAM users

## Compute Services

### Amazon EC2
Elastic Compute Cloud provides virtual machines in the cloud.

#### Key Concepts:
- Instance Types: General Purpose, Compute Optimized, Memory Optimized, Storage Optimized
- User Data Scripts: Used to bootstrap EC2 instances on launch
- Security Groups: Virtual firewalls controlling inbound/outbound traffic
- Purchasing Models: On-Demand, Reserved, Spot, Dedicated Hosts, and Capacity Reservations
- Load Balancing: Elastic Load Balancer (ELB) distributes traffic across multiple instances
- High Availability: Achieved using Auto Scaling Groups and Multi-AZ deployments

## Storage Services

### Amazon EBS
Elastic Block Store provides persistent storage for EC2.

- Volume Types: gp2/gp3 (SSD), io1/io2 (PIOPS), st1/sc1 (HDD)
- Snapshots: Point-in-time backups of volumes
- Multi-Attach: io1/io2 volumes can be mounted to multiple EC2 instances (cluster-aware apps only)

### Amazon EFS
Elastic File System is a scalable NFS for Linux instances.

- Multi-AZ Access: Accessible across AZs with high availability
- Lifecycle Policies: Automatically move files between Standard and Infrequent Access tiers

### Amazon S3
Scalable object storage for any type of file.

- Buckets and Objects
- Lifecycle Rules: Automate archival and deletion
- Versioning: Preserve, retrieve, and restore every version of every object
- Event Notifications: Trigger workflows on object events
- Static Website Hosting: Serve frontend apps directly from S3

## Networking

### VPC (Virtual Private Cloud)
Logical network isolation within AWS.

- Subnets: Public or private networks within a VPC
- Route Tables: Define traffic routing rules
- Internet Gateway & NAT Gateway: Manage internet access for subnets
- Security Groups vs NACLs: Stateful vs stateless packet filtering

### Elastic Load Balancing (ELB)
Distributes traffic across resources.

#### Types:
- Application Load Balancer (ALB): Layer 7 (HTTP/HTTPS)
- Network Load Balancer (NLB): Layer 4 (TCP/UDP)
- Gateway Load Balancer (GWLB): Layer 3
- Health Checks: Monitor and remove unhealthy targets

## Serverless Services

### AWS Lambda
Run code without provisioning servers.

- Triggers: S3, API Gateway, DynamoDB Streams, EventBridge, etc.
- Resource Limits: Up to 15 minutes execution, 10 GB ephemeral storage
- IAM Role Integration: Grants temporary credentials to access other AWS resources
- Environment Variables: Used for config and secrets

### API Gateway
Expose REST/HTTP/WebSocket APIs.

- Stages and Deployments: Manage API versions
- Throttling and Caching: Control traffic and improve performance
- Authentication: IAM, Lambda authorizers, and Cognito

### Step Functions
Coordinate workflows across services using state machines.

## Application Integration

### Amazon SQS
Managed message queue service.

- Standard vs FIFO Queues
- Dead-Letter Queues (DLQ)
- Visibility Timeout and Long Polling

### Amazon SNS
Pub/Sub messaging system.

- Topic-Based Architecture
- Multiple Protocol Support: Email, SMS, Lambda, HTTP, SQS
- Message Filtering

## Monitoring & Observability

### Amazon CloudWatch
Centralized monitoring and logging.

- Metrics and Dashboards
- Alarms: Trigger actions based on thresholds
- Logs: Aggregate logs from Lambda, EC2, and other services

### AWS X-Ray
Distributed tracing system.

- Traces and Segments: Analyze latency and dependencies
- Service Map: Visualize microservice architecture

### CloudTrail
Records every API call made in the account for auditing and compliance.

## Infrastructure as Code

### AWS CloudFormation
Define infrastructure as code using JSON or YAML.

- Stacks and Nested Stacks
- Parameters, Outputs, Mappings
- Change Sets: Preview resource changes before applying them

## Development Tools

### AWS CLI
Command-line interface for AWS.

- Script Automation
- Profile Management
- JSON/YAML Output Formatting

### AWS SDKs
Programmatic interfaces for popular languages.

- Python (Boto3), JavaScript, Java, Go, etc.
- Integrated with IAM for secure access

## Cost and Resource Management

- EC2 Pricing Models: Optimize based on workload predictability
- IPv4 Costs: Starting Feb 2024, public IPv4 has associated costs
- Right-sizing and Auto-scaling: Avoid over-provisioning

## Conclusion

This guide summarizes key AWS concepts—compute, storage, security, networking, serverless, and monitoring—based entirely on the detailed content from Stéphane Maarek's AWS Developer training material. These building blocks form the foundation for scalable, resilient, and secure cloud-native applications.

---

*Note: This guide is based on publicly available AWS documentation and best practices. For the most up-to-date information, always refer to the official AWS documentation.* 