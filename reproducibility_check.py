import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebook.ipynb")
    ap.add_argument("--submission", default="output/submission.csv")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=-1, help="detik per sel; -1 = tanpa batas")
    args = ap.parse_args()

    nb, sub = Path(args.notebook), Path(args.submission)
    if not nb.exists():
        print(f"Notebook tidak ditemukan: {nb}")
        return 1

    work = Path(tempfile.mkdtemp(prefix="repro_"))
    reference = None
    if sub.exists():
        reference = sha256(sub)
        shutil.copy(sub, work / "reference.csv")
        print(f"Referensi (submisi yang ada)  : {reference}")
    else:
        print("PERINGATAN: tidak ada submisi referensi; hanya membandingkan antar-run.")

    hashes = []
    for k in range(1, args.runs + 1):
        if sub.exists():
            sub.unlink()  # hapus agar file lama tidak bisa lolos secara tidak sengaja
        print(f"\n=== Run {k}/{args.runs}: menjalankan notebook dari nol ===", flush=True)
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               f"--ExecutePreprocessor.timeout={args.timeout}",
               "--output", f"run{k}.ipynb", "--output-dir", str(work), str(nb)]
        proc = subprocess.run(cmd, env={**os.environ})
        if proc.returncode != 0 or not sub.exists():
            print(f"GAGAL: notebook error atau {sub} tidak dibuat pada run {k}.")
            return 1
        h = sha256(sub)
        shutil.copy(sub, work / f"run{k}.csv")
        hashes.append(h)
        print(f"Run {k} SHA-256               : {h}")

    ok_runs = len(set(hashes)) == 1
    ok_ref = reference is None or reference == hashes[0]
    if reference is not None:
        shutil.copy(work / "reference.csv", sub)  # kembalikan submisi referensi
    print("\n--- Hasil ---")
    print(f"Antar-run identik            : {ok_runs}")
    if reference is not None:
        print(f"Sama dengan submisi referensi: {ok_ref}")
    print("REPRODUCIBLE" if (ok_runs and ok_ref) else "TIDAK REPRODUCIBLE")
    return 0 if (ok_runs and ok_ref) else 1


if __name__ == "__main__":
    sys.exit(main())