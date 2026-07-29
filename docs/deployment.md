# Deployment and operations

## Current topology

The live service is [checkers.upsidedownatlas.com](https://checkers.upsidedownatlas.com).
The HTTPS apex, `upsidedownatlas.com`, has its own origin certificate and permanently
redirects to the checkers hostname while preserving the request path.

```text
browser
  -> Cloudflare proxy, edge TLS/DDoS controls/cache
  -> Hetzner firewall (80/443 accepted only from Cloudflare networks)
  -> Caddy 2.10.2, automatic origin certificate, security headers
  -> 127.0.0.1:8765 Python policy/game service
```

The origin is an x86-64 Ubuntu 26.04 LTS Hetzner instance in Germany with one vCPU, 2 GB RAM, and no GPU. Inference is CPU-only. Germany is entirely adequate for this turn-based game; the measured model work is milliseconds, so transatlantic network latency affects feel more than inference but does not justify the roughly $25 hosting premium for the current audience.

Cloudflare Tunnel is **not** used in this release. The available scoped token could edit DNS but lacked Tunnel Write, so deployment uses proxied DNS plus a Cloudflare-only origin firewall. This fallback is explicit and does not broaden direct web access to the origin.

Both hostnames must remain in the Caddy configuration. Cloudflare error 525 means
the edge could not negotiate TLS with the origin; in this deployment, an apex-only
525 was resolved by adding the apex site block so Caddy could obtain its separate
certificate before issuing the redirect.

## Container controls

The application image uses pinned Node and Python base-image digests, builds the Vite client, installs the hash-locked CPU-only runtime, and runs as UID/GID 10001. The container is read-only, drops all capabilities, has `no-new-privileges`, a 64-process cap, 1 GB memory and one CPU, a 16 MiB temporary filesystem, log rotation, init/reaping, graceful shutdown, and a startup health check.

Caddy is independently pinned, read-only, limited to 256 MB/0.5 CPU, and retains only `NET_BIND_SERVICE`. The app model and sidecar are mounted read-only. Startup fails closed on absent, unreadable, malformed, non-finite, or checksum-mismatched weights.

## Host controls

- Dedicated `mlapp` operator with key-only SSH and constrained administrative membership.
- Direct root login, passwords, and keyboard-interactive authentication disabled.
- UFW default-deny inbound; SSH allowed, web allowed only from Cloudflare's published IPv4/IPv6 ranges.
- Docker installed from its official apt repository and supervised by systemd.
- App and proxy use `restart: unless-stopped`; health is checked every 30 seconds.
- Structured application logs normalize game routes so UUIDs are not recorded.

Games are deliberately ephemeral. At most 256 sessions are retained, idle sessions expire after six hours, and a process restart discards every match.

## Release layout

```text
/opt/ml-lab-checkers/
  current -> releases/<release-id>
  releases/<release-id>/
    deploy/checkers/
    models/checkers/policies/
    src/checkers/
    web/checkers/
```

Each production image should have an immutable release tag through `CHECKERS_RELEASE`; do not rely on `latest`.

```bash
release_id=$(date -u +%Y%m%dT%H%M%SZ)
export CHECKERS_RELEASE="$release_id"
docker compose -f deploy/checkers/compose.yaml build --pull checkers-web
docker compose -f deploy/checkers/compose.yaml up -d --wait
curl --fail http://127.0.0.1:8765/api/health
```

Never copy `.secrets`, `.git`, full checkpoints, run state, CUDA environments, or credentials to the host. Model files must be mode `0444` (or otherwise readable by UID 10001 but not writable). The first deployment caught this boundary when a preserved local `0600` mode correctly caused startup to fail.

## Verification checklist

```bash
docker compose -f deploy/checkers/compose.yaml ps
docker stats --no-stream checkers-web checkers-caddy
curl --fail https://checkers.upsidedownatlas.com/api/health
curl --fail https://checkers.upsidedownatlas.com/api/model
```

Also verify a complete human/model exchange, invalid JSON/content type/illegal move responses, mobile layout, security headers, static cache HIT, API/HTML `no-store`, direct-origin timeout, root SSH rejection, container restart, and session loss after restart.

## Rollback

Keep the previous release directory and image tag until the new release passes external smoke tests.

```bash
cd /opt/ml-lab-checkers
sudo ln -sfn "releases/<previous-release>" current
cd current
export CHECKERS_RELEASE='<previous-image-tag>'
docker compose -f deploy/checkers/compose.yaml up -d --no-build --force-recreate --wait
curl --fail http://127.0.0.1:8765/api/health
```

To roll forward, point `current` back to the new release and repeat with its immutable image tag. A rollback is not proven merely by changing the symlink; the container must be recreated, pass health, expose the expected model checksum, and complete a legal move.

## Observability and known limits

Docker health/restart policy, structured JSON logs, bounded log files, resource ceilings, Cloudflare analytics, and host disk checks provide basic operational coverage. There is no account system, persistent game database, external pager, or automated billing alert. The service has concurrency/body/session bounds and Cloudflare DDoS protection, but no per-user application rate limiter. Add one only with a trusted client-IP design if traffic warrants it.

Secrets remain only in ignored local storage and provider credential stores. Cloudflare/API tokens and private SSH keys must never be copied into release directories, environment files, image layers, logs, issues, or GitHub Actions.
