#!/bin/bash

# Vast.AI setup


# Define variables
RUNNER_VERSION="2.320.0"
RUNNER_ARCH="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCH}"
RUNNER_DIR="/opt/actions-runner"
GH_USER="gh"
CONTAINER_LABEL=$(cat ~/.vast_containerlabel)
INSTANCE_ID=${CONTAINER_LABEL#"C."}

if [ -z "$TMUX_SESSION_NAME" ]; then
  echo "Error: TMUX_SESSION_NAME is not set in the environment."
  exit 1
fi

if [ -z "$INSTANCE_ID" ]; then
  echo "Error: INSTANCE_ID is not defined."
  exit 1
fi

# Update and install dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y curl tar wget coreutils git tree python3-pip

pip install vastai

vastai destroy instance $INSTANCE_ID


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
RUNNER_ALLOW_RUNASROOT=1 ./config.sh --unattended --url https://github.com/$GITHUB_REPOSITORY --token $GITHUB_ACTIONS_TOKEN --labels gpu --disableupdate --ephemeral

echo "export PATH=\$PATH:/usr/local/julia/bin/" >> ~/.bashrc
echo "export JULIA_NUM_THREADS=16" >> ~/.bashrc
echo "export JULIA_SLOW_PROGRESS_BAR=true">> ~/.bashrc

echo $(realpath ~/.bashrc)
cat ~/.bashrc

# Mark the runner as ready
touch /root/READY

# Setup the runner package
RUNNER_ALLOW_RUNASROOT=1 ./run.sh

if [ -z "$" ]; then
  echo "Usage: $0 <session-name>"
  exit 1
fi

while tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; do
  sleep 1  # Wait for 1 second before checking again
done

echo "Session '$TMUX_SESSION_NAME' has ended."

# Kill myself
vastai destroy instance $INSTANCE_ID

