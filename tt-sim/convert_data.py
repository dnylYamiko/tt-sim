import numpy as np
from pathlib import Path


def convert_folder_pos3_to_t7(input_dir, output_dir, dt=0.1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found under {input_dir}")

    converted = 0
    skipped = 0

    for file in files:
        pos = np.load(file)

        if pos.ndim != 2 or pos.shape[1] != 3:
            print(f"Skipping {file}: expected (T, 3), got {pos.shape}")
            skipped += 1
            continue

        T = pos.shape[0]
        t = np.arange(T, dtype=np.float32) * dt
        vel = np.gradient(pos, dt, axis=0)

        traj = np.column_stack([t, pos, vel]).astype(np.float32)

        out_file = output_dir / file.name
        np.save(out_file, traj)

        print(f"Converted {file} -> {out_file}, shape {traj.shape}")
        converted += 1

    print(f"\nConverted: {converted}")
    print(f"Skipped: {skipped}")


convert_folder_pos3_to_t7(
    input_dir="data/kienzle_120k",
    output_dir="data/kienzle_120k_t7",
    dt=0.1
)