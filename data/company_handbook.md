# Acme Corp Engineering Handbook & Security Policies

## 1. Remote Work & Equipment Policy
Employees at Acme Corp may work remotely from anywhere within their hiring country. Acme provides each engineer with a standard equipment package:
- 16-inch MacBook Pro (Apple Silicon) or ThinkPad X1 Carbon with Ubuntu LTS.
- A home office stipend of $1,500 USD replenished every 24 months.
- A recurring internet reimbursement of up to $80 USD per month.

All hardware must be enrolled in Acme's Mobile Device Management (MDM) system within 48 hours of receipt. Unenrolled devices are strictly prohibited from accessing internal code repositories or VPN gateways.

## 2. Infrastructure Access & Authentication
Engineers access staging and production infrastructure strictly through Teleport and single sign-on (SSO) with FIDO2 WebAuthn hardware security keys (e.g., YubiKey). 
- Static SSH keys and shared credentials are prohibited across all environments.
- Production database read-only access requires a peer-approved access request valid for a maximum duration of 4 hours.
- Production write or administrative access requires dual authorization (two engineering leads) and generates an immutable audit log sent to our SIEM cluster.

## 3. Incident Management & On-Call Rotation
On-call rotations run weekly, beginning and ending on Tuesdays at 10:00 AM UTC.
There are three severity tiers for incidents:
- **Sev-1 (Critical):** Core user-facing functionality is down or customer data integrity is at risk. MTTR target is under 30 minutes. The on-call primary must acknowledge the pager duty alarm within 5 minutes.
- **Sev-2 (Degraded):** Non-critical features impacted or partial service degradation affecting less than 10% of users. Target MTTR is under 2 hours. Response time SLA is 15 minutes.
- **Sev-3 (Minor):** Internal developer tooling disruptions or low-priority edge case defects. Addressed during business hours.

Every Sev-1 and Sev-2 incident requires a blameless post-mortem document completed in Notion within 48 hours of resolution.

## 4. Code Review & Deployment Standards
All code committed to `main` must pass through automated CI pipelines and receive at least one code review approval from an owner defined in the repository's `CODEOWNERS` file.
- Direct pushes to `main` and force-pushes are permanently disabled via branch protection rules.
- Deployments to production must be gated behind canary analysis running for a minimum of 15 minutes with automated rollback if 5xx error rate exceeds 0.05%.
