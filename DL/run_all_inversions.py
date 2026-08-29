"""
run_all_inversions.py

Iterate over timestamp subdirectories (format YYYYMMDD_HHMM or
YYYYMMDD_HHMMSS) in a base directory and run the DEM inversion
implemented in fullBP.py for each timestamp.

All hyperparameters accepted by fullBP.py are forwarded verbatim via
the `--hparams` argument to avoid duplicating fullBP.py's argparse.

Example:
        python run_all_inversions.py /path/with/timestamps /output/path \
                --hparams "--fitfn lp_nnls --tolfac 1.4 --decimate 4 --parallel -1"

Behavior:
    1. Finds subfolders named YYYYMMDD_HHMM or YYYYMMDD_HHMMSS.
    2. For each timestamp runs:
                python fullBP.py <subdir> <output>/<timestamp>.npz <hparams>
    3. Skips failing timestamps and reports them at the end.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time


# Accepts YYYYMMDD_HHMM or YYYYMMDD_HHMMSS
PATRON_TIMESTAMP = re.compile(r"^\d{8}_\d{4,6}$")

# By default assume fullBP.py lives one directory above this script (project root)
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fullBP.py"
)


def encontrarCarpetasTimestamp(baseDir):
    """Return a sorted list of subdirectories in ``baseDir`` whose
    names match the timestamp pattern produced by dlAIA_v2.py.
    """
    carpetas = []
    for nombre in os.listdir(baseDir):
        ruta = os.path.join(baseDir, nombre)
        if os.path.isdir(ruta) and PATRON_TIMESTAMP.match(nombre):
            carpetas.append(nombre)
    carpetas.sort()
    return carpetas


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Run the DEM inversion (fullBP.py) for all downloaded timestamps"
    )
    parser.add_argument("srcBase", help="directory containing one subfolder per timestamp")
    parser.add_argument("targetBase", help="directory where output .npz files are saved")
    parser.add_argument(
        "--hparams", default="",
        help=(
            "hyperparameters to pass to fullBP.py as a single quoted string."
            " Example: \"--fitfn lp_nnls --tolfac 1.4 --decimate 4 --parallel -1\""
        )
    )
    parser.add_argument(
        "--fullbp", default=DEFAULT_PATH,
        help=(
            "path to fullBP.py (default: the one next to this script)"
        )
    )
    parser.add_argument(
        "--visTargetBase", default="",
        help=(
            "optional: base directory where per-timestamp HTML visualizations are created"
        )
    )
    parser.add_argument(
        "--seguirSiFalla", action="store_true", default=True,
        help="continue with other timestamps if one fails (default behavior)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    if not os.path.isfile(args.fullbp):
        print("ERROR: fullBP.py not found at: %s" % args.fullbp)
        print("       Use --fullbp <path> to specify its location.")
        sys.exit(1)

    if not os.path.exists(args.targetBase):
        os.makedirs(args.targetBase)

    # Split the hyperparameter string into an argv-style list, respecting quotes
    hparams = shlex.split(args.hparams)

    timestamps = encontrarCarpetasTimestamp(args.srcBase)
    print("Found %d timestamps to process in %s" % (len(timestamps), args.srcBase))
    if hparams:
        print("Hyperparameters to use:", " ".join(hparams))

    fallidos = []
    for i, ts in enumerate(timestamps):
        srcDir = os.path.join(args.srcBase, ts)
        targetFile = os.path.join(args.targetBase, "%s.npz" % ts)

        comando = [sys.executable, args.fullbp, srcDir, targetFile] + hparams

        if args.visTargetBase:
            visDir = os.path.join(args.visTargetBase, ts)
            comando += ["--visTarget", visDir]

        print("\n[%d/%d] Inverting %s ..." % (i + 1, len(timestamps), ts))
        print("  Command:", " ".join(comando))

        tic = time.time()
        resultado = subprocess.run(comando)
        toc = time.time()

        if resultado.returncode != 0:
            print("  -> Failed %s (exit code %d)" % (ts, resultado.returncode))
            fallidos.append(ts)
            if not args.seguirSiFalla:
                break
        else:
            print("  -> Done in %.1f s" % (toc - tic))

    exitosos = len(timestamps) - len(fallidos)
    print("\nProcess finished. %d/%d timestamps succeeded." % (exitosos, len(timestamps)))
    if fallidos:
        print("Failed:", fallidos)