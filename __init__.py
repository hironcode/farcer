import sys
import os
from pathlib import Path

print("="*30)
print(f"Activated: {__file__}")
print("="*30)


root = os.path.join("/".join(__file__.split("/")[:-1]))
if root in sys.path:
    print(f"{root} already in sys.path")
else:
    sys.path.insert(0, root)
    print(f"Added {root} to sys.path")