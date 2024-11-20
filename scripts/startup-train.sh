#! /bin/bash

# Vast.AI setup

sudo apt-get update -y
sudo apt-get install -y wget coreutils git
git clone https://github.com/${REPO_NAME}.git

cd ${REPO_NAME}/lib/${SUBPACKAGE} && ./scripts/train.sh

if [ $? -eq 0 ]; then
    touch ~/SUCCESS
    echo "Command succeeded. ~/SUCCESS created."
else
    touch ~/FAILED
    echo "Command failed. ~/FAILED created."
fi

