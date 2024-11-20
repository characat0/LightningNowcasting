#! /bin/bash

# Vast.AI setup


# Define variables
RUNNER_VERSION="2.320.0"
RUNNER_ARCH="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCH}"
RUNNER_DIR="/opt/actions-runner"
GH_USER="gh"

# Update and install dependencies
echo "Updating system and installing dependencies..."
sudo apt update
sudo apt install -y curl tar wget coreutils git

# Create the 'gh' user if it doesn't exist
if ! id -u "$GH_USER" &>/dev/null; then
    echo "Creating user '$GH_USER'..."
    sudo useradd --create-home --shell /bin/bash "$GH_USER"
else
    echo "User '$GH_USER' already exists."
fi

# Create runner directory
echo "Setting up runner directory..."
mkdir -p "$RUNNER_DIR"
chown "$GH_USER:$GH_USER" "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Download the runner package
echo "Downloading GitHub Actions runner version $RUNNER_VERSION..."
sudo -u "$GH_USER" curl -o "$RUNNER_ARCH" -L "$RUNNER_URL"
echo "93ac1b7ce743ee85b5d386f5c1787385ef07b3d7c728ff66ce0d3813d5f46900  $RUNNER_ARCH" | shasum -a 256 -c
sudo -u "$GH_USER" tar xzf "$RUNNER_ARCH"

# Setup the runner package
sudo -u "$GH_USER" ./config.sh --unattended --url https://github.com/$GITHUB_REPOSITORY --token $GITHUB_ACTIONS_TOKEN --labels gpu

# Mark the runner as ready
touch /root/READY

# Setup the runner package
sudo -u "$GH_USER" ./run.sh


# repo=$(echo ${REPO_NAME} | cut -d'/' -f2)

# useradd -m gh



# curl -o actions-runner-osx-x64-2.320.0.tar.gz -L

# while :; do
#   # Run sha256sum and capture the output
#   output=$(cd ${repo}/lib/${SUBPACKAGE} && sha256sum -c data/SHA256SUMS 2>&1)

#   # Check if the output contains "FAILED open" or "FAILED read"
#   if echo "$output" | grep -qE "FAILED open or read"; then
#     echo "Some files are not yet downloaded. Waiting 1 minute..."
#     sleep 60
#   else
#     echo "All files verified successfully!"
#     break
#   fi
# done

# cd ${repo}/lib/${SUBPACKAGE} && chmod +x ./scripts/train.sh  && ./scripts/train.sh

# if [ $? -eq 0 ]; then
#     touch ~/SUCCESS
#     echo "Command succeeded. ~/SUCCESS created."
# else
#     touch ~/FAILED
#     echo "Command failed. ~/FAILED created."
# fi

