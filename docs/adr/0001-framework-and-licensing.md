# ADR-0001: Detector framework and licensing boundary

- Status: Accepted for M0, open for production decision
- Date: 2026-07-30

## Context

The original project copied part of Ultralytics 8.3.234 into `yolo_source/ultralytics/` and modified internal parser, block, and loss files. The copied package declares AGPL-3.0 while the repository root declares MIT. The copied package was also incomplete: the required `ultralytics/data` package was absent.

The legacy weight depends on the custom model parser and modules, so deleting the fork before a verified replacement exists would make the baseline non-reproducible.

## Decision

During M0:

- retain the legacy source only as a pinned baseline implementation;
- restore the missing `ultralytics/data` package from upstream tag `v8.3.234`;
- record upstream version, license, weight hash, and local modifications;
- prevent new business logic from importing Ultralytics result types directly;
- put all new system code behind framework-neutral detector interfaces.

Before production distribution, the project must either obtain a suitable commercial license or migrate to a framework with acceptable licensing. Two independent training stacks will not be maintained long-term.

## Consequences

- M0 can reproduce the historical detector before model migration.
- The repository remains unsuitable for unreviewed closed-source redistribution.
- New video, tracking, geometry, event, and service code can survive a future detector-framework migration.
