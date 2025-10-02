#!/usr/bin/env bash


# Check if k4_local_repo function is defined, if not source utilities.sh
if ! declare -f k4_local_repo > /dev/null; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    . "$script_dir/utilities.sh"
fi

cmake_build_system=$(which ninja > /dev/null 2>&1 && echo "-GNinja")
ccache_config=$(which ccache > /dev/null 2>&1 && echo "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")

# Clone a git repository if the target directory doesn't already exist
function clone_if_not_exists() {
    local repo_url="$1"
    shift

    local folder_name=$(basename "$repo_url" .git)

    if [ -d "$folder_name" ]; then
        echo "$folder_name folder already exists, skipping clone"
    else
        git clone "$repo_url" "$@"
    fi
}

# Clone all dependencies
function clone_dependencies() {
    clone_if_not_exists https://github.com/key4hep/k4ActsTracking
    clone_if_not_exists https://github.com/tmadlener/acts --branch build-gnn-plugin-no-cuda
    clone_if_not_exists https://github.com/rusty1s/pytorch_scatter
}

function build_pytorch_scatter() {
    cmake -B pytorch_scatter/build -S pytorch_scatter \
        -DWITH_CUDA=OFF \
        -DWITH_PYTHON=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/pytorch_scatter/install \
        ${cmake_build_system} ${ccache_config}

    cmake --build pytorch_scatter/build --target install
}

function build_acts() {
    cmake -B acts/build -S acts \
        -DACTS_BUILD_PLUGIN_GNN=ON \
        -DACTS_GNN_ENABLE_CUDA=OFF \
        -DACTS_GNN_ENABLE_ONNX=ON \
        -DACTS_GNN_ENABLE_TORCH=ON \
        -DACTS_BUILD_PLUGIN_DD4HEP=ON \
        -DACTS_BUILD_PLUGIN_JSON=ON \
        -DACTS_USE_SYSTEM_NLOHMANN_JSON=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/acts/install \
        ${cmake_build_system} ${ccache_config}

    cmake --build acts/build --target install
}

function build_k4actstracking() {
    cmake -B k4ActsTracking/build -S k4ActsTracking \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/k4ActsTracking/install \
        ${cmake_build_system} ${ccache_config}

    cmake --build k4ActsTracking/build --target install
}

function build_dependencies() {
    build_pytorch_scatter
    export CMAKE_PREFIX_PATH=$(pwd)/pytorch_scatter/install:$CMAKE_PREFIX_PATH

    build_acts
    cd acts && k4_local_repo
    cd ../

    build_k4actstracking
    cd k4ActsTracking && k4_local_repo
    cd ../
}

clone_dependencies
build_dependencies
