#!/usr/bin/env sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if k4_local_repo function is defined, if not source utilities.sh
if ! declare -f k4_local_repo > /dev/null; then
    . "$script_dir/utilities.sh"
fi

curr_dir=$(pwd)

export CMAKE_PREFIX_PATH=$(realpath ${script_dir}/../pytorch_scatter/install):$CMAKE_PREFIX_PATH
cd ${script_dir}/../acts && k4_local_repo
cd ${script_dir}/../k4ActsTracking && k4_local_repo
cd ${curr_dir}
