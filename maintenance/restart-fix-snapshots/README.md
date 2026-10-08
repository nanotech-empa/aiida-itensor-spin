# Preserved restart fixes

These snapshots preserve work from the current container without installing
either fix into production. The workflow fix also has its own published branch,
`fix/monotonic-spectral-segment-index`, in this repository.

The AiiDA core change and regression tests are stored as `aiida-core.patch`
against the base in `snapshot.json`. A Git bundle additionally preserves the
actual core commit and branch. Its required base is available from the public
`aiidateam/aiida-core` repository. In a checkout containing that base:

```bash
git bundle verify /path/to/aiida-core-restart-fix.bundle
git fetch /path/to/aiida-core-restart-fix.bundle \
  fix/restart-child-after-limit-reset:fix/restart-child-after-limit-reset
```

The core development base was 281 commits behind fetched upstream main on
2026-10-08. Refresh and reproduce the change against current upstream before
opening a core PR. The historical October 5 regression evidence does not
establish readiness on current upstream. Production still imports AiiDA 2.8.0
from site-packages.

`historical_v5/` preserves two older L101 launchers for their operational record.
They refer to older codes, checkpoints and UUIDs; do not use them to resume the
current paused L101 calculation.
