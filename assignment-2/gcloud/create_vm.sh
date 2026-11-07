#!/usr/bin/env bash
# Creates a Google Cloud VM with an NVIDIA GPU (L4 or T4) for Assignment 2.
#
# Usage: ./create_vm.sh PROJECT_ID [l4|t4]
#
# The second argument picks the GPU type (default: l4). See the "GPU options"
# section below for the machine each one uses.
#
# What it does:
#   1. Exits if the VM already exists. Otherwise, deletes any network left over
#      from a previous run (see teardown in common.sh).
#   2. Creates a private network for the VM, plus a firewall rule that allows SSH.
#   3. Goes region by region looking for a zone that has the GPU available, and
#      creates the VM in the first one that works. GPUs are in high demand, so
#      some zones will say they have none available; the script just moves on.
#   4. Waits while the VM installs the NVIDIA driver and the Python dependencies
#      (see setup_vm.sh), then prints how to connect.
#
# A few Google Cloud terms used below:
#   - project: the container for everything you create in Google Cloud. Billing
#     and permissions are attached to it. You pass its ID to this script.
#   - region / zone: a region is a geographic location (e.g., us-central1, in
#     Iowa). Each region has several zones (us-central1-a, -b, -c), which are
#     separate data centers. A VM lives in one zone.
#   - gcloud: Google Cloud's command-line tool. "gcloud compute ..." commands
#     manage VMs and networking (Compute Engine).
set -u
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 PROJECT_ID [l4|t4|cpu]"
  exit 1
fi
PROJECT=$1
GPU_CHOICE=${2:-l4}
# Loads NET (network name), VM (VM name), log, and teardown
source "$(dirname "$0")/common.sh"

# --- GPU options ---
# Machine type = how many CPUs and how much memory the VM gets.
#   MACHINE      the machine type.
#   GPU          the GPU's name in Google Cloud. Empty for a VM without a GPU.
#   ACCELERATOR  extra option for "gcloud compute instances create" that attaches
#                the GPU. Empty when the machine type already comes with one,
#                or when there is no GPU.
#   REGIONS      regions to try, in order. Regions without the GPU are skipped,
#                so it's fine to list extra ones.
# To try another GPU, add a case here.
case "$GPU_CHOICE" in
  l4)
    # g2-standard-4 has 4 virtual CPUs, 16 GB of memory, and always comes with
    # one NVIDIA L4 GPU (with 24 GB of GPU memory).
    MACHINE=g2-standard-4
    GPU=nvidia-l4
    ACCELERATOR=""
    REGIONS="us-central1 us-east1 us-east4 us-west1 us-west4 us-south1"
    ;;
  t4)
    # n1-standard-4 has 4 virtual CPUs and 15 GB of memory. N1 machines don't
    # come with a GPU, so we attach one NVIDIA T4 GPU (with 16 GB of GPU memory).
    MACHINE=n1-standard-4
    GPU=nvidia-tesla-t4
    ACCELERATOR="--accelerator=type=$GPU,count=1"
    REGIONS="us-central1 us-east1 us-east4 us-west1 us-west2 us-west3 us-west4"
    ;;
  cpu)
    # n1-standard-4 (4 virtual CPUs, 15 GB of memory) with no GPU. Training is
    # much slower, but these machines are almost always available.
    MACHINE=n1-standard-4
    GPU=""
    ACCELERATOR=""
    REGIONS="us-central1 us-east1 us-east4 us-west1 us-west2 us-west3 us-west4 us-south1"
    ;;
  *)
    echo "Unknown GPU type '$GPU_CHOICE'. Use l4, t4, or cpu."
    exit 1
    ;;
esac
log "Using ${GPU:-no GPU} on $MACHINE."

# Names are prefixed with $NET so they can't collide with other resources in
# the project (firewall names are project-wide; subnet/router names per region).
FIREWALL=$NET-allow-ssh-iap
SUBNET=$NET-subnet
ROUTER=$NET-router
NAT=$NET-nat

# Make $PROJECT the default project, so later gcloud commands act on it without
# needing a --project option each time.
log "Setting active project to $PROJECT..."
gcloud config set project "$PROJECT" || exit 1

# Refuse to run if the VM already exists (running or stopped). teardown would
# delete the router/NAT and firewall rule it relies on, cutting off its SSH and
# internet access, and then fail to delete the network because the VM uses it.
#   --filter="name=$VM"                 keep only the VM with our VM's name.
#   --format="value(zone.basename())"   print only its zone, without a header.
log "Checking whether $VM already exists..."
EXISTING_ZONE=$(gcloud compute instances list --filter="name=$VM" --format="value(zone.basename())")
if [ -n "$EXISTING_ZONE" ]; then
  log "$VM already exists in $EXISTING_ZONE. To use it, start it (if stopped) with:"
  log "  gcloud compute instances start $VM --zone=$EXISTING_ZONE"
  log "To replace it with a new VM, first delete it (all files on its disk are lost):"
  log "  $(dirname "$0")/stop_vm.sh $PROJECT --delete"
  exit 1
fi

teardown

# --- global pieces ---

# Create a VPC (Virtual Private Cloud) network: a private network for our VM,
# separate from anything else in the project.
#   --subnet-mode=custom  don't automatically create a subnet in every region;
#                         we create one ourselves, only where we need it.
log "Creating network $NET..."
gcloud compute networks create $NET --subnet-mode=custom || exit 1

# Create a firewall rule. By default a new network blocks all incoming traffic;
# this rule lets SSH connections in.
#   --network=$NET                     the network the rule applies to.
#   --allow=tcp:22                     allow TCP port 22, which is SSH.
#   --source-ranges=35.235.240.0/20    only from these IP addresses, which belong
#                                      to Google's IAP (Identity-Aware Proxy).
#                                      The VM has no public IP address, so you
#                                      can't reach it from the internet; instead,
#                                      "gcloud compute ssh --tunnel-through-iap"
#                                      asks Google to forward your connection
#                                      from these addresses, after checking that
#                                      your Google account is allowed in.
log "Creating firewall rule $FIREWALL (SSH via IAP)..."
gcloud compute firewall-rules create $FIREWALL --network=$NET \
  --allow=tcp:22 --source-ranges=35.235.240.0/20 || exit 1

# List every GPU type offered in every zone, and keep the zones that offer $GPU.
#   --format="value(name,zone.basename())"  print just the GPU name and the zone
#                                           name (e.g., "nvidia-l4 us-central1-a"),
#                                           one per line, without a table header.
# The awk command then keeps lines whose first column is $GPU and prints the zone.
# (gcloud's own --filter option on accelerator names turned out to be unreliable,
# so we filter in the shell instead.)
# Without a GPU, any zone will do, so we list all zones instead.
if [ -n "$GPU" ]; then
  log "Looking up zones that offer $GPU GPUs..."
  ZONES=$(gcloud compute accelerator-types list --format="value(name,zone.basename())" \
    | awk -v gpu="$GPU" '$1 == gpu {print $2}')
else
  log "Looking up zones..."
  ZONES=$(gcloud compute zones list --format="value(name)")
fi

# Wait for setup_vm.sh, which runs on the VM, to finish. setup_vm.sh prints lines
# starting with "SETUP:" as it goes; we read them from the VM's serial console.
wait_for_setup() {
  local zone=$1 last=""
  log "Waiting for the VM to install the NVIDIA driver and Python dependencies"
  log "(usually 10-15 minutes; the VM reboots once)..."
  # Check every 20 seconds, up to 90 times (30 minutes)
  for _ in $(seq 1 90); do
    sleep 20
    # Read the VM's serial console: the text output the VM prints while it boots
    # and runs its startup script, like a screen attached to the machine.
    #   --zone=$zone  the zone the VM is in.
    # We keep only the last "SETUP:" line, i.e., the latest progress message.
    local status=$(gcloud compute instances get-serial-port-output $VM --zone=$zone 2>/dev/null \
      | grep -oE "SETUP: .*" | tail -1)
    case "$status" in
      "SETUP: ready"*)  log "$status"; return 0 ;;
      "SETUP: failed"*) log "$status"; return 1 ;;
    esac
    if [ -n "$status" ] && [ "$status" != "$last" ]; then
      log "$status"
      last=$status
    else
      log "Still waiting..."
    fi
  done
  log "Setup not finished after 30 minutes. Check on the VM with:"
  log "  gcloud compute ssh $VM --zone=$zone --tunnel-through-iap -- sudo journalctl -u google-startup-scripts"
  return 1
}

print_connect_info() {
  local zone=$1
  echo
  echo "To connect to the VM (it has no external IP, so SSH goes through IAP):"
  echo "  gcloud compute ssh $VM --zone=$zone --project=$PROJECT --tunnel-through-iap"
  echo "The first time, gcloud creates an SSH key for you and may ask for a passphrase."
  echo "Once connected, the Python venv at /opt/venv is already active."
  echo
  echo "When you're done, stop the VM so it doesn't keep billing:"
  echo "  $(dirname "$0")/stop_vm.sh $PROJECT            # stop (keeps the disk)"
  echo "  $(dirname "$0")/stop_vm.sh $PROJECT --delete   # delete the VM, disk, and network"
}

# --- per-region sweep ---
i=1
for region in $REGIONS; do
  log "=== $region ==="
  # Each region's subnet gets its own address range: 192.168.1.x, 192.168.2.x, ...
  range=192.168.$i.0/24
  i=$((i+1))

  zones=$(echo "$ZONES" | grep "^$region-")
  if [ -z "$zones" ]; then
    log "No ${GPU:-usable} zones in $region; moving on."
    continue
  fi
  log "${GPU:-Usable} zones in $region: $(echo $zones)"

  # Create a subnet: the part of our network that exists in this region. A VM
  # must be placed in a subnet, and gets its private IP address from it.
  #   --network=$NET                      the network the subnet belongs to.
  #   --region=$region                    the region it lives in.
  #   --range=$range                      the private IP addresses VMs in it can
  #                                       get (e.g., 192.168.1.0/24 means
  #                                       192.168.1.0 to 192.168.1.255).
  #   --enable-flow-logs                  record a log of network connections
  #                                       to and from VMs in this subnet, which
  #                                       you can view in Cloud Logging.
  #   --logging-flow-sampling=1.0         log all connections (1.0 = 100%), not
  #                                       just a sample.
  #   --enable-private-ip-google-access   let VMs without a public IP address
  #                                       still reach Google services (e.g.,
  #                                       Cloud Storage) directly.
  log "Creating subnet $SUBNET ($range) in $region..."
  gcloud compute networks subnets create $SUBNET --network=$NET --region=$region \
    --range=$range --enable-flow-logs --logging-flow-sampling=1.0 \
    --enable-private-ip-google-access \
    || { log "Subnet creation failed in $region; skipping region."; continue; }

  # Create a Cloud Router in this region. On its own it does nothing for us,
  # but Cloud NAT (next command) must be attached to a router.
  #   --network=$NET    the network the router serves.
  #   --region=$region  the region it lives in.
  log "Creating Cloud Router $ROUTER in $region..."
  gcloud compute routers create $ROUTER --network=$NET --region=$region \
    || { log "Router creation failed in $region; skipping region."; continue; }

  # Create a Cloud NAT (Network Address Translation) gateway. Our VM has no public
  # IP address, so on its own it can't reach the internet. NAT lets it make
  # outgoing connections (e.g., to download the NVIDIA driver and Python packages)
  # while still not accepting any incoming connections from the internet.
  #   --router=$ROUTER                  the router to attach it to.
  #   --region=$region                  the region it lives in.
  #   --auto-allocate-nat-external-ips  let Google pick the public IP address(es)
  #                                     that outgoing traffic appears to come from.
  #   --nat-all-subnet-ip-ranges        apply to every VM in every subnet of the
  #                                     network in this region.
  log "Creating Cloud NAT $NAT in $region..."
  gcloud compute routers nats create $NAT --router=$ROUTER --region=$region \
    --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges \
    || { log "NAT creation failed in $region; skipping region."; continue; }

  for z in $zones; do
    # Create the VM ("instance"). This fails if the zone has no free GPUs (or,
    # rarely, machines) right now, in which case we try the next zone.
    #   --zone=$z                     the zone to create it in.
    #   --machine-type=$MACHINE       CPUs, memory, and GPU (see MACHINE above).
    #   $ACCELERATOR                  attaches the GPU, if the machine type
    #                                 doesn't come with one (see GPU options).
    #   --subnet=$SUBNET              the subnet to put it in.
    #   --no-address                  don't give it a public IP address, so it
    #                                 can't be reached directly from the internet.
    #   --maintenance-policy=TERMINATE
    #                                 when Google needs to do maintenance on the
    #                                 physical machine, stop the VM instead of
    #                                 moving it while running. VMs with GPUs
    #                                 can't be moved, so this is required.
    #   --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud
    #                                 the operating system to install: the latest
    #                                 Ubuntu 24.04 image, published by Canonical
    #                                 in its "ubuntu-os-cloud" project.
    #   --metadata-from-file=...      attach files to the VM as metadata (named
    #                                 values the VM can read once it's running):
    #                                   startup-script: setup_vm.sh. The VM runs
    #                                     whatever is under this special name
    #                                     every time it boots.
    #                                   python-deps: install_python_deps.sh, which
    #                                     setup_vm.sh downloads and runs.
    #   --metadata=VmDnsSetting=ZonalOnly
    #                                 register the VM's internal name in its own
    #                                 zone's DNS only, so name lookups don't
    #                                 depend on other regions being up.
    #   --boot-disk-size=50GB         size of the VM's disk.
    #   --quiet                       don't ask for confirmation.
    log "Trying to create $VM in $z..."
    gcloud compute instances create $VM --zone=$z --machine-type=$MACHINE $ACCELERATOR \
      --subnet=$SUBNET --no-address --maintenance-policy=TERMINATE \
      --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
      --metadata-from-file=startup-script="$(dirname "$0")/setup_vm.sh",python-deps="$(dirname "$0")/../install_python_deps.sh" \
      --metadata=VmDnsSetting=ZonalOnly \
      --boot-disk-size=50GB --quiet \
      && { log "SUCCESS: $VM created in $z"; wait_for_setup $z; status=$?; print_connect_info $z; exit $status; }
    log "Could not create $VM in $z; trying next zone."
  done
done
log "No capacity found in any region tried; cleaning up the network."
teardown
exit 1
