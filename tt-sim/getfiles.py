from pathlib import Path
import numpy as np

root = Path("C:\\Users\\zp2105023\\pythonprojects\\fancy_gym\\fancy_gym\\tt-sim\\data\\kienzle_50k")

files = sorted(root.rglob("*.npy")) + sorted(root.rglob("*.npz"))

print("Files found:", len(files))

for f in files[:20]:
    print("\nFile:", f)

    if f.suffix == ".npy":
        arr = np.load(f, allow_pickle=True)
        print("shape:", arr.shape)

    elif f.suffix == ".npz":
        data = np.load(f)
        print("keys:", data.files)
        for k in data.files:
            value = data[k]
            print(k, value.shape if hasattr(value, "shape") else value)