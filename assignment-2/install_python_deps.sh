#!/usr/bin/env bash
# Installs the Python dependencies for Assignment 2 into the active virtual environment.
# The venv must be created with --system-site-packages so that ROS 2 Python packages
# (rclpy, cv_bridge, urdf_parser_py, scipy, ...) remain visible inside it.
set -e

if [ -z "${VIRTUAL_ENV}" ]; then
    echo "No Python virtual environment is active. Create and activate one first, e.g.:"
    echo "  cd <path-to-your-ros2-workspace>"
    echo "  python3 -m venv --system-site-packages .venv"
    echo "  source .venv/bin/activate"
    exit 1
fi

# Versions are pinned so that everyone runs the same code as the course staff during the semester.
TORCH_VERSION=2.14.1
# numpy must match the version that ROS 2 Jazzy's system packages (cv_bridge and friends) are built
# against, because the venv's site-packages are on the PYTHONPATH when running the ROS 2 nodes and tests.
# Every pip3 command below repeats the pin so that no package upgrades numpy as a dependency.
NUMPY_VERSION=1.26.4
# matplotlib is also pinned to the version that ROS 2 Jazzy installs through apt.
MATPLOTLIB_VERSION=3.6.3
# setuptools must satisfy both torch (>=77.0.3) and colcon-core (<80). 
SETUPTOOLS_VERSION=79.0.1
TENSORBOARD_VERSION=2.21.0
GDOWN_VERSION=6.4.2

# PyTorch. The default wheel works on CPU-only machines and on NVIDIA GPUs from Turing (RTX 20xx) onward.
pip3 install "torch==${TORCH_VERSION}" "numpy==${NUMPY_VERSION}" "setuptools==${SETUPTOOLS_VERSION}"
# For older Pascal GPUs (e.g., the GTX 1070/1080 in the lab), comment out the line above and use
# the CUDA 12.6 build instead. CUDA 13 dropped Pascal support.
# pip3 install "torch==${TORCH_VERSION}" "numpy==${NUMPY_VERSION}" "setuptools==${SETUPTOOLS_VERSION}" --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple

# Other dependencies
pip3 install "tensorboard==${TENSORBOARD_VERSION}" "matplotlib==${MATPLOTLIB_VERSION}" \
    "gdown==${GDOWN_VERSION}" "numpy==${NUMPY_VERSION}" "setuptools==${SETUPTOOLS_VERSION}"
