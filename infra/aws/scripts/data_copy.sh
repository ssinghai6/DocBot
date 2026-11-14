#!/usr/bin/env bash
# DOCBOT-2516: copy the Chroma store (/data/chroma) from the Railway volume to
# EFS, byte for byte (ADR infra/aws/adr/0001-chroma-data.md, option A).
#
#   data_copy.sh export <out.tgz>   Railway -> local tarball (needs `railway` CLI, linked to the project)
#   data_copy.sh import <in.tgz>    local tarball -> S3 -> one-off ECS task -> EFS /data/chroma
#
# Integrity: the tarball's sha256 is taken inside the Railway container, checked
# locally after the transfer, and checked again inside the ECS task before it
# extracts. Any mismatch aborts before /data is touched.
#
# export env:  RAILWAY_SERVICE (optional, the backend service name)
#              SRC_CHROMA_DIR  (default /data/chroma, the prod CHROMA_DIR on Railway)
# import env:  AWS_CLUSTER, AWS_SERVICE  ECS cluster + API service (task def, subnets, SGs are copied from it)
#              S3_URI                    s3://bucket/prefix/chroma-<date>.tgz (bucket with default encryption)
#              FORCE=1                   replace a non-empty /data/chroma (old copy kept as chroma.bak-<ts>)
#
# Run export only after the write freeze (no uploads in flight, cutover T-0) so
# the SQLite files are quiescent. Run import only with the ECS service scaled to
# 0, so nothing has the store open while it is swapped in.
set +x
set -euo pipefail
umask 077

die() { echo "data_copy: $*" >&2; exit 2; }
sha256() { python3 -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(b) for b in iter(lambda: f.read(1<<20), b"")]; print(h.hexdigest())' "$1"; }

cmd="${1:-}"; file="${2:-}"
[ -n "$file" ] || die "usage: data_copy.sh export|import <file.tgz>"

case "$cmd" in
export)
  command -v railway >/dev/null || die "railway CLI not installed"
  src="${SRC_CHROMA_DIR:-/data/chroma}"
  rw=(railway ssh ${RAILWAY_SERVICE:+--service "$RAILWAY_SERVICE"} --)
  echo "1/3 tarring $src inside the Railway container..."
  remote_sha="$("${rw[@]}" sh -c "tar czf /tmp/chroma-export.tgz -C '$(dirname "$src")' '$(basename "$src")' && sha256sum /tmp/chroma-export.tgz" \
    | tr -d '\r' | grep -Eo '^[0-9a-f]{64}' | tail -1)"
  [ -n "$remote_sha" ] || die "no sha256 from the Railway container"
  echo "2/3 streaming (base64)..."
  "${rw[@]}" sh -c "base64 /tmp/chroma-export.tgz; rm -f /tmp/chroma-export.tgz" \
    | python3 -c 'import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))' > "$file"
  local_sha="$(sha256 "$file")"
  echo "3/3 verifying: remote=$remote_sha local=$local_sha"
  [ "$remote_sha" = "$local_sha" ] || die "sha256 mismatch after transfer; re-run export"
  tar tzf "$file" | head -3
  echo "OK: $file ($(du -h "$file" | cut -f1)). Contains user document text: keep it private, delete after import."
  ;;
import)
  command -v aws >/dev/null || die "aws CLI not installed"
  for v in AWS_CLUSTER AWS_SERVICE S3_URI; do [ -n "${!v:-}" ] || die "$v is not set"; done
  [ -f "$file" ] || die "no such file: $file"
  tar tzf "$file" | grep -q '^chroma/' || die "tarball must contain chroma/ at its root"
  sha="$(sha256 "$file")"

  svc="$(aws ecs describe-services --cluster "$AWS_CLUSTER" --services "$AWS_SERVICE" --output json)"
  read -r desired running taskdef < <(python3 -c 'import json,sys; s=json.load(sys.stdin)["services"][0]; print(s["desiredCount"], s["runningCount"], s["taskDefinition"])' <<<"$svc")
  [ "$desired" = "0" ] && [ "$running" = "0" ] || die "scale $AWS_SERVICE to 0 first (desired=$desired running=$running)"
  container="$(aws ecs describe-task-definition --task-definition "$taskdef" --query 'taskDefinition.containerDefinitions[0].name' --output text)"
  netcfg="$(python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["services"][0]["networkConfiguration"]))' <<<"$svc")"

  echo "1/3 uploading to S3 (sha256 $sha)..."
  aws s3 cp --only-show-errors "$file" "$S3_URI"
  url="$(aws s3 presign "$S3_URI" --expires-in 900)"

  # Runs as appuser via docker-entrypoint.sh (which chowns /data first).
  read -r -d '' PY <<'PYEOF' || true
import hashlib, os, shutil, sys, tarfile, time, urllib.request
tgz, dst, tmp = "/tmp/chroma.tgz", "/data/chroma", "/data/.chroma-incoming"
urllib.request.urlretrieve(os.environ["CHROMA_TGZ_URL"], tgz)
h = hashlib.sha256()
with open(tgz, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
if h.hexdigest() != os.environ["CHROMA_TGZ_SHA256"]:
    sys.exit("sha256 mismatch: download corrupted, /data untouched")
if os.path.isdir(dst) and os.listdir(dst) and os.environ.get("FORCE") != "1":
    sys.exit("refusing: /data/chroma is not empty (re-run with FORCE=1)")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
with tarfile.open(tgz) as t:
    t.extractall(tmp, filter="data")
if os.path.exists(dst):
    os.rename(dst, "%s.bak-%d" % (dst, time.time()))
os.rename(os.path.join(tmp, "chroma"), dst)
os.rmdir(tmp)
n = sum(len(fs) for _, _, fs in os.walk(dst))
print("chroma restored: %d files, sha256 %s" % (n, h.hexdigest()))
PYEOF
  overrides="$(CONTAINER="$container" URL="$url" SHA="$sha" PY="$PY" python3 -c '
import json, os
env = [{"name": "CHROMA_TGZ_URL", "value": os.environ["URL"]}, {"name": "CHROMA_TGZ_SHA256", "value": os.environ["SHA"]},
       {"name": "FORCE", "value": os.environ.get("FORCE", "")}]
print(json.dumps({"containerOverrides": [{"name": os.environ["CONTAINER"], "command": ["python", "-c", os.environ["PY"]], "environment": env}]}))')"
  unset url

  echo "2/3 running one-off import task..."
  task="$(aws ecs run-task --cluster "$AWS_CLUSTER" --task-definition "$taskdef" --launch-type FARGATE \
    --network-configuration "$netcfg" --overrides "$overrides" --query 'tasks[0].taskArn' --output text)"
  unset overrides
  aws ecs wait tasks-stopped --cluster "$AWS_CLUSTER" --tasks "$task"
  code="$(aws ecs describe-tasks --cluster "$AWS_CLUSTER" --tasks "$task" --query 'tasks[0].containers[0].exitCode' --output text)"
  echo "3/3 task exit code: $code (logs: CloudWatch, stream for task ${task##*/})"
  [ "$code" = "0" ] || { echo "data_copy: import FAILED" >&2; exit 1; }
  echo "OK: /data/chroma replaced. Scale $AWS_SERVICE back up and run smoke.sh."
  ;;
*) die "usage: data_copy.sh export|import <file.tgz>" ;;
esac
