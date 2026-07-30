# Safety, privacy, license, and operations review

Status: engineering checklist complete; formal approvals remain open.

- Safety: human confirmation is mandatory; no autonomous flight/rescue command is implemented; degraded estimates
  are explicit; safe-stop and abort criteria are documented.
- Privacy: raw video may contain identifiable people and precise locations. A site-specific lawful basis, notice,
  access control, encryption, retention/deletion, export, and incident policy is required before collection.
- License: the repository includes an Ultralytics AGPL-3.0 code boundary and third-party notices. Commercial or
  network-service distribution requires qualified legal review and either AGPL compliance or an appropriate
  commercial licensing arrangement. This document is not legal approval.
- Operations: signed packages, immutable versions, canary rollback, local evidence, health monitoring, fault drills,
  backup/restore, credential rotation, and named on-call ownership are required.

The M8 submission needs independent named reviewers and evidence for all four domains. Self-generated unit tests do
not constitute approval.
