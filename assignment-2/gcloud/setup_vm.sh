#!/usr/bin/env bash
# Startup script for the VM created by create_vm.sh. You don't run this file
# yourself: create_vm.sh attaches it to the VM as metadata under the special
# name "startup-script", and Google's software on the VM runs it as root
# (administrator) every time the VM boots.
#
# On first boot it installs the NVIDIA driver, creates a Python venv at
# $VENV, runs install_python_deps.sh in it (also attached as metadata, under
# the name "python-deps"), and reboots so the driver loads. Later boots check
# that PyTorch can see the GPU.
#
# Everything this script prints goes to the VM's serial console (its boot-time
# text output). create_vm.sh reads the console and follows the "SETUP:" lines
# to show you progress. On the VM, you can see the full output with:
#   sudo journalctl -u google-startup-scripts

# This file is created once the first-boot install finishes, so later boots
# know to skip it.
MARKER=/var/lib/vm-setup-done
VENV=/opt/venv

# Check whether the VM has an NVIDIA GPU (create_vm.sh's "cpu" option makes one
# without). Every PCI device lists its maker's ID in /sys; 0x10de is NVIDIA's.
if grep -qs 0x10de /sys/bus/pci/devices/*/vendor; then
  HAS_GPU=1
else
  HAS_GPU=0
fi

if [ -f $MARKER ] && [ $HAS_GPU = 0 ]; then
  if $VENV/bin/python -c "import torch" 2>/dev/null; then
    echo "SETUP: ready (no GPU; PyTorch will run on the CPU)"
    exit 0
  fi
  echo "SETUP: failed: packages installed but PyTorch cannot be imported."
  exit 1
fi

if [ -f $MARKER ]; then
  GPU=$($VENV/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" 2>/dev/null)
  if [ -n "$GPU" ]; then
    echo "SETUP: ready (PyTorch sees $GPU; driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader))"
    exit 0
  fi
  if nvidia-smi >/dev/null 2>&1; then
    echo "SETUP: failed: nvidia-smi works but PyTorch cannot see the GPU (driver may be too old for this torch build)."
  else
    echo "SETUP: failed: packages installed but nvidia-smi does not work."
  fi
  exit 1
fi

fail() {
  echo "SETUP: failed while $*; see 'sudo journalctl -u google-startup-scripts' on the VM."
  exit 1
}

export DEBIAN_FRONTEND=noninteractive
# Wait for the dpkg lock instead of failing if unattended-upgrades holds it
APT="apt-get -y -o DPkg::Lock::Timeout=600"

# Finish any package install that was interrupted (e.g., by a hard reset while
# Ubuntu's automatic updates were running); otherwise apt-get refuses to run.
dpkg --configure -a

# ubuntu-drivers is Ubuntu's tool for picking and installing the right driver
# for the hardware it finds (here, the L4 or T4 GPU).
$APT update || fail "updating the package list"
if [ $HAS_GPU = 1 ]; then
  echo "SETUP: installing NVIDIA driver..."
  $APT install ubuntu-drivers-common && ubuntu-drivers install \
    || fail "installing the NVIDIA driver"
else
  echo "SETUP: no NVIDIA GPU found; skipping the driver."
fi

# python3-numpy and python3-matplotlib are the same versions install_python_deps.sh
# pins (Ubuntu 24.04's, which ROS 2 Jazzy also uses). The venv sees them through
# --system-site-packages, so pip doesn't try to build matplotlib 3.6.3 from source
# (it has no prebuilt wheel for Python 3.12, and the build fails).
echo "SETUP: creating Python venv at $VENV..."
$APT install python3-venv python3-numpy python3-matplotlib \
  && python3 -m venv --system-site-packages $VENV \
  || fail "creating the Python venv"
# Activate the venv in every login shell
echo "source $VENV/bin/activate" > /etc/profile.d/venv.sh

# Download install_python_deps.sh from the VM's metadata. Every Google Cloud VM
# can ask the "metadata server" (metadata.google.internal) for information about
# itself, including the metadata create_vm.sh attached. The "Metadata-Flavor:
# Google" header is required by the server.
echo "SETUP: installing Python dependencies from install_python_deps.sh (this takes a while)..."
curl -fsS -H "Metadata-Flavor: Google" -o /tmp/install_python_deps.sh \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/python-deps \
  || fail "reading install_python_deps.sh from instance metadata"
(source $VENV/bin/activate && bash /tmp/install_python_deps.sh) \
  || fail "running install_python_deps.sh"

touch $MARKER
echo "SETUP: all installed; rebooting..."
reboot
