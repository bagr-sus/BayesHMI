import ray
import os
import sys

# Locate the real internal module
import ray._private.node
import ray._private.worker
import ray._private.workers
import ray._private.utils

# Parse args from shell script
port = int(sys.argv[1])
num_cpus = int(sys.argv[2])

# Get SCRATCHDIR from env
scratch = os.environ["SCRATCHDIR"]

# Patch the function that constructs the session dir
def short_session_dir(temp_dir: str) -> str:
    return os.path.join(temp_dir, "r")

# Use your actual SCRATCHDIR here
ray.init(
    _temp_dir=os.environ["SCRATCHDIR"],
    num_cpus=num_cpus,
    logging_level="debug",
)

# Replace the internal function used during ray.init()
ray._private.worker._global_node.session_name = "bruh"