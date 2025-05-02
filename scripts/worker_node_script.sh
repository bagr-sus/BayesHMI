#!/bin/bash
set -x

head_address=$1
temp_dir=$2
echo "tempdir: $temp_dir"

source venv/bin/activate
ray start --address=$head_address --temp-dir=$temp_dir
