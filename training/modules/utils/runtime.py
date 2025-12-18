import sys
import signal
import warnings
import yaml # pip install pyyaml

from training.modules.utils.seed import set_random_seed

# Ctrl+Cによる強制終了時の挙動
def handler(signum, frame):
    ans = input("\nDo you want to exit?(y/n): ").lower()
    if ans == "y":
        sys.exit(0)
    else:
        print("Continuing...")

def setup_runtime(seed):
    warnings.filterwarnings(
        "ignore",
        message="The PyTorch API of nested tensors is in prototype stage and will change in the near future."
    )
    signal.signal(signal.SIGINT, handler)
    set_random_seed(seed)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
