# Shared settings and helpers, sourced by create_vm.sh and stop_vm.sh.
NET=vpc-network
VM=my-gpu-vm

# Print a message with the current time, e.g. "[10:12:53] Creating network..."
log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- teardown: if the network exists, remove it and everything attached ---
# Google Cloud won't delete a network while anything still uses it, so we delete
# the pieces attached to it first (routers, firewall rules, subnets) and the
# network last. A VM attached to the network also blocks deletion; stop_vm.sh
# --delete removes the VM before calling this.
#
# The "list" commands below use two options:
#   --filter="network~/$NET$"  keep only resources on our network. Each resource
#                              records its network as a URL ending in
#                              ".../networks/vpc-network"; "~" matches it against
#                              a pattern, and "/$NET$" means "ends with /$NET".
#   --format="value(...)"      print only the listed fields (e.g., name and
#                              region), one resource per line, without a table
#                              header, so the shell can loop over them.
# The "delete" commands use --quiet to skip the "Are you sure?" prompt.
teardown() {
  # Look up the network. Its output is hidden; we only care whether it exists.
  log "Checking whether network $NET exists..."
  if ! gcloud compute networks describe $NET >/dev/null 2>&1; then
    log "Network $NET does not exist; nothing to tear down."
    return
  fi
  log "Network $NET exists; tearing it down..."

  # Delete Cloud Routers on the network. Deleting a router also deletes the
  # Cloud NAT attached to it. Routers belong to a region, so list prints
  # "name<TAB>region", which we turn into "name,region" and split apart again
  # with ${r%%,*} (name) and ${r##*,} (region).
  log "Looking for routers on $NET..."
  for r in $(gcloud compute routers list --filter="network~/$NET$" \
               --format="value(name,region)" | tr '\t' ','); do
    log "Deleting router ${r%%,*} in ${r##*,}..."
    gcloud compute routers delete "${r%%,*}" --region="${r##*,}" --quiet
  done

  # Delete firewall rules on the network. Firewall rules aren't tied to a
  # region, so the name is enough.
  log "Looking for firewall rules on $NET..."
  for f in $(gcloud compute firewall-rules list --filter="network~/$NET$" \
               --format="value(name)"); do
    log "Deleting firewall rule $f..."
    gcloud compute firewall-rules delete "$f" --quiet
  done

  # Delete subnets on the network. Like routers, each belongs to a region.
  log "Looking for subnets on $NET..."
  for s in $(gcloud compute networks subnets list --filter="network~/$NET$" \
               --format="value(name,region)" | tr '\t' ','); do
    log "Deleting subnet ${s%%,*} in ${s##*,}..."
    gcloud compute networks subnets delete "${s%%,*}" --region="${s##*,}" --quiet
  done

  # Finally, delete the network itself.
  log "Deleting network $NET..."
  gcloud compute networks delete $NET --quiet || {
    log "Could not delete $NET — a VM is probably still attached. Remove it and rerun."
    exit 1
  }
  log "Teardown complete."
}
