#!/bin/bash
# =============================================================================
# vm/startup_cpu.sh - startup script of the CPU dev/test VM (also used by the guardrail test VM).
# Instance metadata `startup-script`; runs as root on EVERY boot.
#   1. fetch + source startup_common.sh (metadata key provex-startup-common); idle watchdog first
#   2. mount the persistent data disk at /data - mkfs.ext4 ONLY when the disk has no filesystem
#      at all (no data disk attached -> /data lives on the boot disk)
#   3. common setup: packages, uv, user provex, repo in /data/provex (clone + LFS + extract)
#   4. OPTIONAL PostgreSQL 15 with its data dir on /data/pg, tuned for bulk loads
#      (metadata provex-install-postgres=1, i.e. INSTALL_POSTGRES=1 in config.env)
# =============================================================================
set -Eeuo pipefail

mkdir -p /opt/provex
for _i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS -H 'Metadata-Flavor: Google' -o /opt/provex/startup_common.sh \
       'http://metadata.google.internal/computeMetadata/v1/instance/attributes/provex-startup-common'; then
    break
  fi
  sleep 3
done
if [[ ! -s /opt/provex/startup_common.sh ]]; then
  echo "PROVEX-STARTUP-FAILED epoch=$(date +%s) role=cpu reason=metadata-provex-startup-common-missing" > /dev/ttyS0
  exit 1
fi
# shellcheck source=startup_common.sh
source /opt/provex/startup_common.sh

mount_data_disk() {
  local device_name dev uuid rc=0
  device_name="$(md_attr provex-data-device provex-data)"
  dev="/dev/disk/by-id/google-${device_name}"
  mkdir -p /data
  if [[ ! -b $dev ]]; then log "no data disk (${dev}) attached - /data is on the boot disk"; return 0; fi
  if mountpoint -q /data; then log "/data already mounted"; return 0; fi
  if [[ -z "$(lsblk -no FSTYPE "$dev" 2>/dev/null | tr -d '[:space:]')" ]]; then
    blkid -p "$dev" >/dev/null 2>&1 || rc=$?
    if (( rc == 2 )); then        # blkid: 2 = no filesystem signature at all -> brand-new disk
      log "formatting the EMPTY data disk ${dev} as ext4"
      mkfs.ext4 -q -m 0 -L provex-data -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
    else
      log "ERROR: ${dev} shows no FSTYPE but blkid exit=${rc} - refusing to format it"
      return 1
    fi
  fi
  uuid="$(blkid -s UUID -o value "$dev")"
  if ! grep -q "UUID=${uuid}" /etc/fstab; then
    echo "UUID=${uuid} /data ext4 discard,defaults,nofail 0 2" >> /etc/fstab
  fi
  mount /data
  log "mounted ${dev} at /data ($(df -h --output=size,avail /data | tail -n 1 | xargs))"
}

install_postgres() {  # PostgreSQL 15 (apt.postgresql.org), cluster 15/main with data in /data/pg
  local datadir=/data/pg etcbak=/data/pg-etc conf=/etc/postgresql/15/main
  local secret_dir=/data/pg-secret env_file=/data/pg-secret/provex-db.env
  local db user mem_mb shared cache maint par pw i
  db="$(md_attr provex-pg-db tc_cadet_dataset_db)"
  user="$(md_attr provex-pg-user provex)"
  if [[ ! $db =~ ^[a-z_][a-z0-9_]{0,62}$ || ! $user =~ ^[a-z_][a-z0-9_]{0,62}$ ]]; then
    log "ERROR: invalid provex-pg-db / provex-pg-user"; return 1
  fi
  if ! dpkg -s postgresql-15 >/dev/null 2>&1; then
    log "installing PostgreSQL 15 from apt.postgresql.org"
    install -d /usr/share/postgresql-common/pgdg
    curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc
    # shellcheck source=/dev/null
    . /etc/os-release
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list
    APT_UPDATED=0
    apt_install postgresql-15 postgresql-client-15
  fi
  # the package creates 15/main under /var/lib/postgresql - the cluster belongs on the data disk
  if pg_lsclusters --no-header 2>/dev/null | awk '$1 == 15 && $2 == "main" && $6 ~ /^\/var\/lib\/postgresql/ { f = 1 } END { exit !f }'; then
    log "dropping the default 15/main cluster in /var/lib (re-created on /data/pg)"
    pg_dropcluster --stop 15 main
  fi
  if [[ ! -f $conf/postgresql.conf ]]; then
    if [[ -f $datadir/PG_VERSION ]]; then
      [[ -d $etcbak ]] || { log "ERROR: ${datadir} holds a cluster but its config backup ${etcbak} is missing"; return 1; }
      log "re-registering the existing cluster in ${datadir} (new boot disk)"
      mkdir -p "$conf"
      cp -a "${etcbak}/." "${conf}/"
      [[ "$(stat -c %U "$datadir")" == postgres ]] || chown -R postgres:postgres "$datadir"
    else
      log "initdb: new cluster 15/main in ${datadir}"
      pg_createcluster --datadir="$datadir" --locale=C.UTF-8 --encoding=UTF8 15 main
    fi
  fi
  mem_mb=$(awk '/^MemTotal:/ { print int($2 / 1024) }' /proc/meminfo)
  shared=$(( mem_mb / 4 )); cache=$(( mem_mb * 3 / 4 )); maint=$(( mem_mb / 8 ))
  if (( maint > 2048 )); then maint=2048; fi
  par=$(( $(nproc) / 2 )); if (( par < 1 )); then par=1; fi
  mkdir -p "${conf}/conf.d"
  cat > "${conf}/conf.d/90-provex-bulkload.conf" <<EOF
# Written by startup_cpu.sh at every boot (sized for ${mem_mb} MB RAM): bulk-load tuning.
# synchronous_commit=off may lose the last few hundred ms of commits on a crash, never corrupts.
listen_addresses = 'localhost'
shared_buffers = ${shared}MB
effective_cache_size = ${cache}MB
maintenance_work_mem = ${maint}MB
work_mem = 64MB
wal_buffers = 64MB
max_wal_size = 16GB
min_wal_size = 1GB
checkpoint_timeout = 30min
checkpoint_completion_target = 0.9
wal_compression = on
synchronous_commit = off
random_page_cost = 1.1
effective_io_concurrency = 200
max_parallel_maintenance_workers = ${par}
max_parallel_workers_per_gather = ${par}
EOF
  chown -R postgres:postgres "$conf"
  mkdir -p "$etcbak"
  cp -a "${conf}/." "${etcbak}/"          # lets a re-created VM re-register the cluster
  systemctl daemon-reload
  systemctl enable postgresql@15-main >/dev/null 2>&1 || true
  systemctl restart postgresql@15-main
  for i in $(seq 1 30); do pg_isready -q -h localhost && break; sleep 1; done
  install -d -m 0750 -o root -g "$PROVEX_USER" "$secret_dir"
  if [[ ! -f $env_file ]]; then
    pw="$(openssl rand -hex 16)"
    printf 'PGHOST=localhost\nPGPORT=5432\nPGDATABASE=%s\nPGUSER=%s\nPGPASSWORD=%s\n' "$db" "$user" "$pw" > "$env_file"
    chown "root:${PROVEX_USER}" "$env_file"
    chmod 0640 "$env_file"
  fi
  pw="$(sed -n 's/^PGPASSWORD=//p' "$env_file")"
  if ! (cd /tmp && runuser -u postgres -- psql -qtAc "SELECT 1 FROM pg_roles WHERE rolname = '${user}'") | grep -q 1; then
    (cd /tmp && runuser -u postgres -- psql -v ON_ERROR_STOP=1 -qc "CREATE ROLE ${user} LOGIN CREATEDB PASSWORD '${pw}'")
  fi
  if ! (cd /tmp && runuser -u postgres -- psql -qtAc "SELECT 1 FROM pg_database WHERE datname = '${db}'") | grep -q 1; then
    (cd /tmp && runuser -u postgres -- createdb -O "$user" "$db")
  fi
  printf 'localhost:5432:*:%s:%s\n' "$user" "$pw" > "${PROVEX_HOME}/.pgpass"
  chown "${PROVEX_USER}:${PROVEX_USER}" "${PROVEX_HOME}/.pgpass"
  chmod 0600 "${PROVEX_HOME}/.pgpass"
  log "PostgreSQL 15 ready: db=${db} user=${user}, data ${datadir}, credentials ${env_file} + ~${PROVEX_USER}/.pgpass"
  PROVEX_STATUS_EXTRA="postgres=ok"
}

provex_begin "$(md_attr provex-role cpu)"
mount_data_disk
common_setup
if [[ "$(md_attr provex-install-postgres 0)" == 1 ]]; then
  install_postgres
else
  log "PostgreSQL not requested (INSTALL_POSTGRES=0)"
fi
provex_done
