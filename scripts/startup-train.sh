#! /bin/bash

# Vast.AI setup

sudo apt-get update -y
sudo apt-get install -y wget coreutils git
git clone https://github.com/${REPO_NAME}.git

touch ~/READY

repo=$(echo ${REPO_NAME} | cut -d'/' -f2)

while :; do
  # Run sha256sum and capture the output
  output=$(cd ${repo}/lib/${SUBPACKAGE} && sha256sum -c data/SHA256SUMS 2>&1)

  # Check if the output contains "FAILED open" or "FAILED read"
  if echo "$output" | grep -qE "FAILED open or read"; then
    echo "Some files are not yet downloaded. Waiting 1 minute..."
    sleep 60
  else
    echo "All files verified successfully!"
    break
  fi
done

cd ${repo}/lib/${SUBPACKAGE} && chmod +x ./scripts/train.sh  && ./scripts/train.sh

if [ $? -eq 0 ]; then
    touch ~/SUCCESS
    echo "Command succeeded. ~/SUCCESS created."
else
    touch ~/FAILED
    echo "Command failed. ~/FAILED created."
fi

