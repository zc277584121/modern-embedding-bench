# Milvus SINDI phase-two isolation and recovery checklist

This checklist applies only to the isolated correctness fixture. It does not authorize native,
100k, or 1M performance cells. The research-only and no-publish gates remain closed.

## Fail-closed deployment order

1. Validate the frozen predeclaration and source attestation by their externally supplied SHA-256
   identities.
2. Capture the Docker Registry tag index and the pinned linux/amd64 child manifest as raw bytes.
3. Capture and validate a read-only prelaunch host/Docker baseline.
4. Require the declared container, network, bind root, collection prefix, and loopback ports to be
   absent before creating any Story resource.
5. Pull only the pinned digest with `--platform linux/amd64`; record the image ID and RepoDigests.
6. Create one labeled Story network and one labeled Story container. Require `--pull never`,
   loopback-only port publication, the frozen cpuset, equal memory and memory-swap caps, and the
   declared PID cap.
7. Run only the deterministic twelve-row fixture. Formal workloads remain forbidden.

## Fixture recovery

- Preserve the incomplete private evidence directory before recovery.
- List collections and require an exact allowlist containing only the expected fixture collection.
- A partial fixture collection may be dropped only after that exact allowlist check. Never use a
  wildcard or prefix-wide deletion.
- Re-run into a new evidence directory. Never overwrite an incomplete evidence directory.
- If either index description does not report the requested algorithm, stop immediately and keep
  the formal gate closed.
- A fixture passes only when both algorithms report `Finished`, all rows are indexed, load state is
  `Loaded`, every query segment is `Sealed` with a positive index ID, and the growing segment count
  is zero. Raw server metrics must also report zero compaction executor threads, queue depth, used
  compaction slots, pending tasks, and net executing tasks.

## Container recovery

- Capture Story-container inspect, health, and logs under the gitignored private results tree before
  any recovery action. Logs must never be copied into tracked artifacts.
- Verify the exact container name, full container ID, pinned image digest, and Story labels before
  starting, removing, or recreating the Story container.
- A failed Story container may be removed only while it is not running. Preserve the bind data root
  and Story network, then recreate the same single instance from the pinned digest.
- Do not use Docker prune, Compose project-wide operations, name globs, or commands that target any
  non-Story container, network, volume, or image.
- Never stop, restart, execute into, write to, or clean up a pre-existing service.

## Bind-root hardening and restart recovery

- Before changing permissions, capture a complete read-only host/Docker baseline and an exact
  `stat` record for the declared bind root. Require mode `0777`, the known inode and device, and the
  exact Story container ID and labels.
- Change only the declared bind root to mode `0711`. Never make it world-writable again as an
  automatic recovery step. Require the inode and device to remain unchanged and prove that the
  world-write bit is absent.
- Gracefully stop only the exact Story container after rechecking its full ID and labels. Require
  both loopback ports to be absent, capture a stopped baseline, and do not proceed if a non-Story
  container, network, volume, or listening endpoint changed.
- Start the same existing container; do not recreate it. Require the full container ID, image ID,
  network ID, bind mount, cpuset, memory/no-swap limits, PID limit, and loopback ports to match the
  pre-stop state, then wait for Docker health to report `healthy`.
- Read the two existing fixture collections without inserting, flushing, rebuilding an index, or
  running a performance workload. Require both load states to be `Loaded`, the reported algorithms
  to remain `SINDI` and `DAAT_MAXSCORE`, all twelve rows to remain indexed in sealed segments, and
  the growing segment count to be zero.
- Repeat the fixed `drop_ratio_search=0` fixture searches and recompute exact float32 CSR IP
  correctness with the frozen tie policy. The post-restart metrics response must directly report
  zero active compaction executor threads and zero queued compaction work. Record the presence or
  absence of optional slot and DataCoord task gauges instead of treating an absent optional gauge
  as a numeric zero.
- Capture a post-hardening baseline and require the three-way running/stopped/running isolation
  comparison to pass before the formal gate may report ready. This readiness does not execute or
  independently authorize a native, 100k, or 1M performance cell.

## Deferred cleanup after external approval

Cleanup is intentionally not performed at this checkpoint. Before later cleanup, validate the exact
Story labels and resource IDs again. Remove only the two fixture collections, then the Story
container, then the Story network. The bind data root may be archived or deleted only with an
explicit path check and separate approval. No named Docker volume is owned by this Story.

## Isolation regression

The post-fixture comparison ignores only the declared Story container, Story network, and two
loopback listening endpoints. It requires every non-Story container identity, image, runtime state,
port mapping, mount set, and network set to match; every non-Story Docker network to match; the full
Docker volume list to match; and every non-Story listening endpoint to match.
