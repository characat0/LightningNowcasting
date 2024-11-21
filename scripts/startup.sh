#!/bin/bash

# Vast.AI setup


# Define variables
RUNNER_VERSION="2.320.0"
RUNNER_ARCH="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCH}"
RUNNER_DIR="/opt/actions-runner"
GH_USER="gh"

# Update and install dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y curl tar wget coreutils git tree

# Remove tmux
touch ~/.no_auto_tmux;


# Create runner directory
echo "Setting up runner directory..."
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Download the runner package
echo "Downloading GitHub Actions runner version $RUNNER_VERSION..."
curl -o "$RUNNER_ARCH" -L "$RUNNER_URL"
echo "93ac1b7ce743ee85b5d386f5c1787385ef07b3d7c728ff66ce0d3813d5f46900  $RUNNER_ARCH" | shasum -a 256 -c
tar xzf "$RUNNER_ARCH"

# Setup the runner package
echo "https://github.com/$GITHUB_REPOSITORY"
RUNNER_ALLOW_RUNASROOT=1 ./config.sh --unattended --url https://github.com/$GITHUB_REPOSITORY --token $GITHUB_ACTIONS_TOKEN --labels gpu --ephemeral --disableupdate

echo "export PATH=\$PATH:/usr/local/julia/bin/" >> ~/.bashrc
echo "export JULIA_NUM_THREADS=16" >> ~/.bashrc

echo $(realpath ~/.bashrc)
cat ~/.bashrc

# Mark the runner as ready
touch /root/READY

# Setup the runner package
RUNNER_ALLOW_RUNASROOT=1 ./run.sh

