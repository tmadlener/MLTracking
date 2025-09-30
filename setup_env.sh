#!/usr/bin/env sh

# Check if k4_local_repo function is defined, if not source utilities.sh
if ! declare -f k4_local_repo > /dev/null; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    . "$script_dir/utilities.sh"
fi

export CMAKE_PREFIX_PATH=$(pwd)/pytorch_scatter/install:$CMAKE_PREFIX_PATH
cd acts && k4_local_repo && cd ..
cd kActsTracking && k4_local_repo && cd ..
cd MLTracking && k4_local_repo && cd ..
