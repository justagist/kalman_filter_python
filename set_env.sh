#! /bin/bash

export KF_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$KF_TEST_PATH:$PYTHONPATH

echo -e "Setting KF_TEST_PATH=$KF_TEST_PATH\n"
