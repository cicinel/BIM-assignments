#!/usr/bin/env bash
# Stops or deletes the VM created by create_vm.sh.
#
# Usage: ./stop_vm.sh PROJECT_ID [--delete]
#
#   Without --delete: stops the VM. A stopped VM is like a computer that's turned
#   off: you no longer pay for its CPUs and GPU, but you still pay a little for
#   its disk, which keeps all your files. Start it again with the command this
#   script prints.
#
#   With --delete: deletes the VM and its disk (all files on it are lost), then
#   deletes the network create_vm.sh made for it. After this, nothing from
#   create_vm.sh is left to pay for.
set -u
usage() {
  echo "Usage: $0 PROJECT_ID [--delete]"
  echo "  --delete  delete the VM, its disks, and the network instead of just stopping the VM"
  exit 1
}
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage
fi
PROJECT=$1
DELETE=false
if [ $# -eq 2 ]; then
  [ "$2" = "--delete" ] || usage
  DELETE=true
fi
# Loads NET (network name), VM (VM name), log, and teardown
source "$(dirname "$0")/common.sh"

# Make $PROJECT the default project, so later gcloud commands act on it without
# needing a --project option each time.
log "Setting active project to $PROJECT..."
gcloud config set project "$PROJECT" || exit 1

# create_vm.sh may have placed the VM in any zone, so look it up. This lists
# the project's VMs:
#   --filter="name=$VM"                 keep only the VM with our VM's name.
#   --format="value(zone.basename())"   print only its zone (e.g.,
#                                       "us-central1-b"), without a table header.
log "Looking up the zone of $VM..."
ZONE=$(gcloud compute instances list --filter="name=$VM" --format="value(zone.basename())")
if [ -z "$ZONE" ]; then
  log "No VM named $VM found in $PROJECT."
  if $DELETE; then
    teardown
    exit 0
  fi
  exit 1
fi
log "Found $VM in $ZONE."

if $DELETE; then
  # Delete the VM.
  #   --zone=$ZONE        the zone the VM is in.
  #   --delete-disks=all  also delete its disks (by default gcloud would only
  #                       delete the boot disk; "all" makes sure nothing is left
  #                       behind to keep billing).
  #   --quiet             don't ask "Are you sure?".
  log "Deleting $VM and its disks..."
  gcloud compute instances delete $VM --zone=$ZONE --delete-disks=all --quiet || exit 1
  log "$VM and its disks deleted."
  teardown
  log "SUCCESS: $VM, its disks, and network $NET deleted."
else
  # Stop (shut down) the VM. Its disk and files are kept.
  #   --zone=$ZONE  the zone the VM is in.
  log "Stopping $VM..."
  gcloud compute instances stop $VM --zone=$ZONE || exit 1
  log "SUCCESS: $VM stopped. Restart it with:"
  log "  gcloud compute instances start $VM --zone=$ZONE"
fi
