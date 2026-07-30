#!/usr/bin/env python3
"""Plot derived .basis radial functions against a reference and/or the exact
STO target from wfc.hsd, to visually judge fit quality.

Raw coefficients from an independently-run GTO fit are not comparable
number-by-number to a reference .basis file (STO-to-GTO fitting is a
non-unique nonlinear least-squares problem - see derive_basis_from_wfc.py's
module docstring). The radial *curve* is what matters, so this script plots
curves rather than diffing numbers. One figure is produced per element, with
one subplot per shell.

Examples:
  # Derived vs reference vs exact target, one element, shown interactively
  python3 visualize_basis.py --wfc wfc.hsd --derived-dir mio-1-1-reproduced \\
      --reference-dir mio-1-1-fit --elements C

  # Every element found in --derived-dir, derived vs target only (no reference)
  python3 visualize_basis.py --wfc ob2-1-1/wfc.hsd --derived-dir ob2-1-1-fit

  # Save one PNG per element instead of showing interactively
  python3 visualize_basis.py --wfc wfc.hsd --derived-dir mio-1-1-reproduced \\
      --reference-dir mio-1-1-fit --outdir plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from derive_basis_from_wfc import (
    L2CHAR,
    gto_contract_radial,
    parse_basis_file,
    parse_wfc,
    target_radial_from_wfc,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def find_elements(derived_dir: str | Path) -> list[str]:
    """List element symbols from every *.basis file in derived_dir."""
    return sorted(p.stem for p in Path(derived_dir).glob("*.basis"))


def collect_shells(
    wfc: dict[str, dict[str, Any]],
    derived_dir: str | Path,
    reference_dir: str | Path | None,
    elem: str,
) -> list[dict[str, Any]]:
    """Gather per-shell plot data for one element.

    Returns a list (one entry per shell present in the derived .basis
    file, sorted by angular momentum) of dicts with keys 'l' (int),
    'derived' (the parse_basis_file shell dict), 'reference' (same, or
    None if reference_dir has no matching file/shell), and 'orb' (the
    wfc.hsd orbital dict for the exact STO target, or None if absent).
    """
    _, derived_shells = parse_basis_file(Path(derived_dir) / f"{elem}.basis")

    reference_shells: dict[int, dict[str, Any]] = {}
    if reference_dir:
        ref_path = Path(reference_dir) / f"{elem}.basis"
        if ref_path.exists():
            _, reference_shells = parse_basis_file(ref_path)

    orbitals_by_l = {orb["l"]: orb for orb in wfc[elem]["orbitals"]}
    return [
        {
            "l": shell_l,
            "derived": derived_shells[shell_l],
            "reference": reference_shells.get(shell_l),
            "orb": orbitals_by_l.get(shell_l),
        }
        for shell_l in sorted(derived_shells)
    ]


def auto_rmax(
        shell: dict[str, Any],
        floor: float = 8.0,
        ceiling: float = 60.0,
        threshold: float = 0.01,
        search_npoints: int = 4000,
    ) -> float:
    """Pick a plot range covering a shell's full radial extent.

    Some fitted shells (e.g. an uncompressed/diffuse d-orbital) stay
    significant far past a fixed 8 bohr window - the wfc.hsd 'Cutoff'
    field is not reliable for this (it's a flat default, not the actual
    decay radius). Instead sample the exact STO target (if available,
    else the derived curve) out to `ceiling` and return the largest r
    where its magnitude is still above `threshold` of its peak, plus a
    10% margin, clamped to at least `floor`.
    """
    if shell["orb"] is not None:
        curve_fn = target_radial_from_wfc(shell["orb"]) # type: ignore
    else:
        der = shell["derived"]
        shell_l = shell["l"]

        def curve_fn(r: np.ndarray) -> np.ndarray:
            return gto_contract_radial(r, der["exponents"], der["coeffs"], shell_l)

    r_scan = np.linspace(1e-6, ceiling, search_npoints)
    abs_curve = np.abs(curve_fn(r_scan))
    peak = abs_curve.max()
    if peak <= 0:
        return floor

    above = np.flatnonzero(abs_curve > threshold * peak)
    last_r = r_scan[above[-1]] if above.size else floor
    return float(max(floor, min(ceiling, last_r * 1.1)))


def plot_shell(ax: Axes, r: np.ndarray, shell: dict[str, Any], elem: str) -> None:
    """Draw one shell's exact target / reference / derived curves onto `ax`."""
    shell_l = shell["l"]

    if shell["orb"] is not None:
        target = target_radial_from_wfc(shell["orb"])(r)
        ax.plot(
            r,
            target,
            color="0.7",
            lw=4,
            alpha=0.6,
            label="exact STO target (wfc.hsd)",
            zorder=1,
        )

    if shell["reference"] is not None:
        ref = shell["reference"]
        ref_curve = gto_contract_radial(r, ref["exponents"], ref["coeffs"], shell_l)
        ax.plot(
            r,
            ref_curve,
            color="tab:blue",
            lw=2,
            label=f"reference ({ref['exponents'].size} prim)",
            zorder=2,
        )

    der = shell["derived"]
    der_curve = gto_contract_radial(r, der["exponents"], der["coeffs"], shell_l)
    ax.plot(
        r,
        der_curve,
        color="tab:red",
        lw=1.5,
        ls="--",
        label=f"derived ({der['exponents'].size} prim)",
        zorder=3,
    )

    ax.axhline(0, color="gray", lw=0.5, zorder=0)
    ax.set_title(f"{elem} {L2CHAR[shell_l]}-shell")
    ax.set_xlabel("r (bohr)")
    ax.set_ylabel("R(r)")
    ax.legend(fontsize=8)


def plot_element(
    wfc: dict[str, dict[str, Any]],
    derived_dir: str | Path,
    reference_dir: str | Path | None,
    elem: str,
    rmax: float | None,
    npoints: int,
) -> Figure:
    """Build one figure for elem, with one subplot per shell.

    Each shell gets its own x-axis range: a diffuse/uncompressed shell
    (e.g. an uncompressed d-orbital that stays significant out to 20-30
    bohr) needs a much wider window than a tightly-confined one, so a
    single shared range for the whole figure either clips the diffuse
    shells or wastes most of the plot on empty space for the compact
    ones. When `rmax` is None (the default), each shell's own range is
    picked automatically via auto_rmax; passing an explicit `rmax`
    forces that same fixed range on every shell, as before.
    """
    shells = collect_shells(wfc, derived_dir, reference_dir, elem)

    fig, axes = plt.subplots(
        1, len(shells), figsize=(5.5 * len(shells), 4.5), squeeze=False
    )
    for ax, shell in zip(axes[0], shells):
        shell_rmax = rmax if rmax is not None else auto_rmax(shell)
        r = np.linspace(1e-6, shell_rmax, npoints)
        plot_shell(ax, r, shell, elem)
    fig.suptitle(elem)
    fig.tight_layout()
    return fig


def main() -> None:
    """CLI entry point: plot each requested element, one figure per element."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--wfc",
        "-w",
        required=True,
        help="Path to wfc.hsd (for the exact STO target overlay)",
    )
    ap.add_argument(
        "--derived-dir",
        "-d",
        required=True,
        help="Directory of derived .basis files to plot",
    )
    ap.add_argument(
        "--reference-dir",
        "-r",
        default=None,
        help="Optional directory of reference .basis files to overlay",
    )
    ap.add_argument(
        "--elements",
        "-e",
        nargs="*",
        default=None,
        help="Elements to plot, e.g. C N O (default: every *.basis file in --derived-dir)",
    )
    ap.add_argument("--rmax", "-x", type=float, default=None, help="Plot range in bohr, applied to every shell. Default: auto-detect per shell from where its curve actually decays, so diffuse/uncompressed shells aren't clipped")
    ap.add_argument("--npoints", "-n", type=int, default=2000)
    ap.add_argument(
        "--outdir",
        "-o",
        default=None,
        help="Save one <elem>.png per element into this directory instead of opening interactive windows",
    )
    args = ap.parse_args()

    if args.outdir:
        matplotlib.use("Agg")

    wfc = parse_wfc(args.wfc)
    elements = args.elements if args.elements else find_elements(args.derived_dir)
    if not elements:
        raise SystemExit(
            f"Nothing to plot: no .basis files found in {args.derived_dir}"
        )

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for elem in elements:
        if elem not in wfc:
            raise KeyError(f"{elem} not present in {args.wfc}")
        fig = plot_element(
            wfc, args.derived_dir, args.reference_dir, elem, args.rmax, args.npoints
        )
        if outdir:
            path = outdir / f"{elem}.png"
            fig.savefig(path, dpi=130)  # type: ignore
            plt.close(fig)
            print(f"Saved {path}")

    if not outdir:
        plt.show()


if __name__ == "__main__":
    main()
