import sys
import os
from pathlib import Path


print("="*30)
print(f"Activated: {__file__}")
print("="*30)


src = os.path.join("/".join(__file__.split("/")[:-1]))
print(f"src: {src}")
if src in sys.path:
    print(f"{src} already in sys.path")
else:
    sys.path.insert(0, src)
    print(f"Added {src} to sys.path")