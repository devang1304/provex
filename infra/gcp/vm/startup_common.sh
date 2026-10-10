# shellcheck shell=bash
# =============================================================================
# vm/startup_common.sh - sourced by startup_cpu.sh / startup_gpu.sh ON THE VM (root, every boot).
# Delivered as instance metadata key `provex-startup-common` (create_*_vm.sh refreshes it).
#
#   provex_begin ROLE  logging (/var/log/provex-startup.log), failure marker, settings from
#                      instance metadata, and FIRST the idle watchdog - so the VM is protected
#                      even if a later step fails
#   common_setup       packages (tmux git git-lfs jq python3-venv ...), uv, gcloud/gsutil check,
#                      user $SSH_USER with passwordless sudo, repo in /data/provex (first boot:
#                      git clone + git lfs pull + extract artifact.tar.gz)
#   provex_done        /run/provex-startup.done + "PROVEX-STARTUP-DONE epoch=..." on the serial
#                      console, which the laptop scripts poll (no ssh needed while booting)
# =============================================================================

MD_URL="http://metadata.google.internal/computeMetadata/v1"
md() { curl -fsS --retry 3 -H 'Metadata-Flavor: Google' "${MD_URL}/$1"; }
md_attr() {  # md_attr KEY [DEFAULT]
  local v
  if v="$(md "instance/attributes/$1" 2>/dev/null)"; then printf '%s' "$v"; else printf '%s' "${2:-}"; fi
}
log() { echo "[provex-startup $(date -u +%H:%M:%S)] $*"; }
serial() { echo "$*" > /dev/ttyS0 2>/dev/null || true; echo "$*"; }

provex_fail() {  # ERR trap: leave a marker the laptop can see, then stop the script
  local line=$1 cmd=$2 e
  e=$(date +%s)
  trap - ERR
  printf 'role=%s\nfailed_epoch=%s\nline=%s\ncommand=%s\n' "${PROVEX_ROLE:-?}" "$e" "$line" "$cmd" > /run/provex-startup.failed
  serial "PROVEX-STARTUP-FAILED epoch=${e} role=${PROVEX_ROLE:-?} line=${line} cmd=${cmd}"
  exit 1
}

provex_begin() {  # provex_begin ROLE
  PROVEX_ROLE=$1
  exec > >(tee -a /var/log/provex-startup.log) 2>&1
  trap 'provex_fail "$LINENO" "$BASH_COMMAND"' ERR
  rm -f /run/provex-startup.done /run/provex-startup.failed
  log "=== PROVEX startup, role=${PROVEX_ROLE}, boot $(cat /proc/sys/kernel/random/boot_id) ==="
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
  PROVEX_USER="$(md_attr provex-ssh-user provex)"
  PROVEX_IDLE_MINUTES="$(md_attr provex-idle-minutes 15)"
  PROVEX_REPO_URL="$(md_attr provex-repo-url "")"
  PROVEX_REPO_BRANCH="$(md_attr provex-repo-branch main)"
  PROVEX_REPO_DIR="$(md_attr provex-repo-dir /data/provex)"
  PROVEX_CLONE_REPO="$(md_attr provex-clone-repo 1)"
  PROVEX_STATUS_EXTRA=""
  if [[ ! $PROVEX_USER =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then log "invalid provex-ssh-user - using provex"; PROVEX_USER=provex; fi
  if [[ ! $PROVEX_IDLE_MINUTES =~ ^[0-9]+$ ]]; then PROVEX_IDLE_MINUTES=15; fi
  if [[ $PROVEX_REPO_DIR != /* ]]; then PROVEX_REPO_DIR=/data/provex; fi
  install_watchdog
}

# ------------------------------------------------------------------ idle watchdog
install_watchdog() {
  local tmp
  tmp="$(mktemp)"
  if md "instance/attributes/provex-watchdog" > "$tmp" 2>/dev/null && [[ -s $tmp ]]; then
    install -m 0755 "$tmp" /usr/local/sbin/provex-watchdog
  else
    log "WARNING: metadata provex-watchdog missing - keeping the installed watchdog (if any)"
  fi
  rm -f "$tmp"
  [[ -x /usr/local/sbin/provex-watchdog ]] || { log "ERROR: no watchdog script"; return 1; }
  cat > /etc/provex-watchdog.defaults <<EOF
# Rewritten at every boot from instance metadata. Put overrides in /etc/provex-watchdog.conf.
IDLE_MINUTES=${PROVEX_IDLE_MINUTES}
EOF
  if [[ ! -f /etc/provex-watchdog.conf ]]; then
    cat > /etc/provex-watchdog.conf <<'EOF'
# Local overrides for provex-watchdog (sourced after /etc/provex-watchdog.defaults), e.g.
# IDLE_MINUTES=30
# LOAD_THRESHOLD=0.5
# GPU_UTIL_THRESHOLD=5
EOF
  fi
  cat > /etc/systemd/system/provex-watchdog.service <<'EOF'
[Unit]
Description=PROVEX idle watchdog (powers the VM off when it is idle)

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/provex-watchdog
EOF
  cat > /etc/systemd/system/provex-watchdog.timer <<'EOF'
[Unit]
Description=Run the PROVEX idle watchdog every 5 minutes

[Timer]
OnBootSec=10min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF
  mkdir -p /var/lib/provex-watchdog
  echo 0 > /var/lib/provex-watchdog/count
  rm -f /var/lib/provex-watchdog/since
  systemctl daemon-reload
  systemctl enable provex-watchdog.timer >/dev/null 2>&1
  systemctl restart provex-watchdog.timer
  log "idle watchdog active: power-off after ${PROVEX_IDLE_MINUTES} idle min (first check 10 min after boot)"
}

# ------------------------------------------------------------------ packages & tools
APT_UPDATED=0
apt_install() {  # apt_install PKG... (waits for the apt lock held by unattended-upgrades on first boot)
  local i
  export DEBIAN_FRONTEND=noninteractive
  if (( APT_UPDATED == 0 )); then
    for i in 1 2 3 4 5; do
      if apt-get -o DPkg::Lock::Timeout=600 update -q; then APT_UPDATED=1; break; fi
      log "apt-get update failed (attempt $i) - retrying in 20 s"
      sleep 20
    done
  fi
  apt-get -o DPkg::Lock::Timeout=600 install -y -q --no-install-recommends "$@"
}

common_packages() {
  local p missing=()
  for p in tmux git git-lfs curl ca-certificates jq python3-venv python3-pip; do
    if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed'; then missing+=("$p"); fi
  done
  if (( ${#missing[@]} )); then
    log "installing: ${missing[*]}"
    apt_install "${missing[@]}"
  else
    log "base packages present"
  fi
}

install_uv() {
  if [[ -x /usr/local/bin/uv ]] || command -v uv >/dev/null 2>&1; then log "uv present: $(uv --version 2>/dev/null)"; return 0; fi
  log "installing uv to /usr/local/bin"
  if curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh; then return 0; fi
  log "WARNING: the uv installer failed - trying pip"
  python3 -m pip install --quiet uv || log "WARNING: could not install uv (setup_python.sh installs it per user)"
}

check_gcs_tools() {
  if command -v gcloud >/dev/null 2>&1; then
    log "gcloud available: $(command -v gcloud) (jobs use 'gcloud storage')"
  elif command -v gsutil >/dev/null 2>&1; then
    log "gsutil available: $(command -v gsutil)"
  else
    log "no gcloud/gsutil - installing the google-cloud-cli snap"
    snap install google-cloud-cli --classic || log "WARNING: no GCS client: job logs and the ledger will not reach the bucket"
  fi
}

# ------------------------------------------------------------------ user
ensure_user() {
  if ! id -u "$PROVEX_USER" >/dev/null 2>&1; then
    if getent passwd 2000 >/dev/null; then useradd -m -s /bin/bash "$PROVEX_USER"
    else useradd -m -u 2000 -s /bin/bash "$PROVEX_USER"; fi
    log "created user ${PROVEX_USER}"
  fi
  PROVEX_HOME="$(getent passwd "$PROVEX_USER" | cut -d: -f6)"
  # jobs end with `sudo shutdown -h now`; the guest agent's google-sudoers group may come later
  printf '%s ALL=(ALL) NOPASSWD:ALL\n' "$PROVEX_USER" > /etc/sudoers.d/90-provex
  chmod 0440 /etc/sudoers.d/90-provex
  if ! visudo -cf /etc/sudoers.d/90-provex >/dev/null; then
    rm -f /etc/sudoers.d/90-provex
    log "WARNING: sudoers drop-in rejected"
  fi
}
as_user() {  # run a command as the provex user with a clean HOME/PATH
  runuser -u "$PROVEX_USER" -- env HOME="$PROVEX_HOME" USER="$PROVEX_USER" LOGNAME="$PROVEX_USER" \
    PATH="/usr/local/bin:/usr/bin:/bin:/snap/bin" "$@"
}

# ------------------------------------------------------------------ repo in /data/provex
setup_repo() {
  local dir=$PROVEX_REPO_DIR
  local branch_args=()
  mkdir -p "$(dirname "$dir")"
  install -d -o "$PROVEX_USER" -g "$PROVEX_USER" "$dir"
  # files written by an earlier VM can belong to another uid (the data disk outlives the VM)
  if [[ -n "$(find "$dir" -maxdepth 1 ! -user "$PROVEX_USER" -print -quit 2>/dev/null)" ]]; then
    log "fixing ownership of ${dir}"
    chown -R "${PROVEX_USER}:${PROVEX_USER}" "$dir"
  fi
  if [[ $PROVEX_CLONE_REPO != 1 || -z $PROVEX_REPO_URL ]]; then
    log "repo clone disabled - code arrives with sync_code.sh / run_job.sh"
    return 0
  fi
  as_user git lfs install --skip-repo >/dev/null 2>&1 || log "WARNING: git lfs install failed"
  if [[ ! -d $dir/.git ]]; then
    if [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
      log "WARNING: ${dir} is not empty but has no .git - not cloning (code arrives via sync_code.sh)"
    else
      if as_user git ls-remote --exit-code --heads "$PROVEX_REPO_URL" "$PROVEX_REPO_BRANCH" >/dev/null 2>&1; then
        branch_args=(--branch "$PROVEX_REPO_BRANCH")
      else
        log "WARNING: branch '${PROVEX_REPO_BRANCH}' is not on ${PROVEX_REPO_URL} - cloning the default branch (sync_code.sh overlays your working tree)"
      fi
      log "cloning ${PROVEX_REPO_URL} ${branch_args[*]} -> ${dir} (incl. Git LFS objects)"
      if ! as_user git clone "${branch_args[@]}" "$PROVEX_REPO_URL" "$dir"; then
        log "WARNING: git clone failed - continuing without the repo"
        find "$dir" -mindepth 1 -delete 2>/dev/null || true
        return 0
      fi
    fi
  fi
  if [[ -d $dir/.git ]]; then
    as_user git -C "$dir" lfs pull || log "WARNING: git lfs pull failed (artifact.tar.gz may be a pointer file)"
  fi
  extract_artifact
}

extract_artifact() {  # artifact.tar.gz (Git LFS, ~139 MB) -> artifact/ (graphs + node map), once
  local dir=$PROVEX_REPO_DIR tgz="${PROVEX_REPO_DIR}/artifact.tar.gz" size
  [[ -f $tgz ]] || { log "no artifact.tar.gz in ${dir}"; return 0; }
  size=$(stat -c %s "$tgz")
  if (( size < 1048576 )); then
    log "WARNING: ${tgz} is only ${size} bytes (Git LFS pointer?) - not extracting"
    return 0
  fi
  if [[ -d $dir/artifact/graph_embeddings && -n "$(ls -A "$dir/artifact/graph_embeddings" 2>/dev/null)" ]]; then
    log "artifact/ already extracted"
    return 0
  fi
  log "extracting artifact.tar.gz ($(( size / 1048576 )) MB) into ${dir}"
  as_user tar -xzf "$tgz" -C "$dir"
}

common_setup() {
  common_packages
  install_uv
  check_gcs_tools
  ensure_user
  setup_repo
}

provex_done() {
  local e
  e=$(date +%s)
  {
    echo "role=${PROVEX_ROLE}"
    echo "finished_epoch=${e}"
    if [[ -n $PROVEX_STATUS_EXTRA ]]; then echo "$PROVEX_STATUS_EXTRA"; fi
  } > /run/provex-startup.done
  serial "PROVEX-STARTUP-DONE epoch=${e} role=${PROVEX_ROLE} ${PROVEX_STATUS_EXTRA}"
  log "=== startup finished ==="
}
