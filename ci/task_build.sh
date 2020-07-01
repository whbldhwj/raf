#!/bin/bash
set -e
set -u
set -o pipefail
source ./ci/env.sh

BUILD_DIR=$1
MAKE_FLAGS=$2

pip3 install cmake

set +e
git apply 3rdparty/tvm_patch/tvm_fix.patch
set -e

mkdir -p $BUILD_DIR
cd $BUILD_DIR && $HOME/.local/bin/cmake .. && make $MAKE_FLAGS && make mnm-cpptest $MAKE_FLAGS && cd ..
