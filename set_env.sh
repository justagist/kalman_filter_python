#! /bin/bash

export KF_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$KF_PATH:$PYTHONPATH

echo -e "Setting KF_PATH=$KF_PATH\n"
