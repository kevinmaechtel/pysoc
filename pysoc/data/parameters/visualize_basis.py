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
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from derive_basis_from_wfc import (
    L2CHAR,
    gto_contract_radial,
    parse_basis_file,
    parse_wfc,
    target_radial_from_wfc,
)


def find_elements(
        derived_dir: str | Path
    ) -> list[str]:
    """List element symbols from every *.basis file in derived_dir."""
    return sorted(p.stem for p in Path(derived_dir).glob('*.basis'))


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
    _, derived_shells = parse_basis_file(Path(derived_dir) / f'{elem}.basis')

    reference_shells: dict[int, dict[str, Any]] = {}
    if reference_dir:
        ref_path = Path(reference_dir) / f'{elem}.basis'
        if ref_path.exists():
            _, reference_shells = parse_basis_file(ref_path)

    orbitals_by_l = {orb['l']: orb for orb in wfc[elem]['orbitals']}
    return [
        {
            'l': shell_l,
            'derived': derived_shells[shell_l],
            'reference': reference_shells.get(shell_l),
            'orb': orbitals_by_l.get(shell_l),
        }
        for shell_l in sorted(derived_shells)
    ]


def plot_shell(
        ax: Axes, 
        r: np.ndarray, 
        shell: dict[str, Any], 
        elem: str
    ) -> None:
    """Draw one shell's exact target / reference / derived curves onto `ax`."""
    shell_l = shell['l']

    if shell['orb'] is not None:
        target = target_radial_from_wfc(shell['orb'])(r)
        ax.plot(r, target, color='0.7', lw=4, alpha=0.6, label='exact STO target (wfc.hsd)', zorder=1)

    if shell['reference'] is not None:
        ref = shell['reference']
        ref_curve = gto_contract_radial(r, ref['exponents'], ref['coeffs'], shell_l)
        ax.plot(r, ref_curve, color='tab:blue', lw=2, label=f'reference ({ref["exponents"].size} prim)', zorder=2)

    der = shell['derived']
    der_curve = gto_contract_radial(r, der['exponents'], der['coeffs'], shell_l)
    ax.plot(r, der_curve, color='tab:red', lw=1.5, ls='--', label=f'derived ({der["exponents"].size} prim)', zorder=3)

    ax.axhline(0, color='gray', lw=0.5, zorder=0)
    ax.set_title(f'{elem} {L2CHAR[shell_l]}-shell')
    ax.set_xlabel('r (bohr)')
    ax.set_ylabel('R(r)')
    ax.legend(fontsize=8)


def plot_element(
    wfc: dict[str, dict[str, Any]],
    derived_dir: str | Path,
    reference_dir: str | Path | None,
    elem: str,
    rmax: float,
    npoints: int,
) -> Figure:
    """Build one figure for elem, with one subplot per shell."""
    shells = collect_shells(wfc, derived_dir, reference_dir, elem)
    r = np.linspace(1e-6, rmax, npoints)

    fig, axes = plt.subplots(1, len(shells), figsize=(5.5 * len(shells), 4.5), squeeze=False)
    for ax, shell in zip(axes[0], shells):
        plot_shell(ax, r, shell, elem)
    fig.suptitle(elem)
    fig.tight_layout()
    return fig


def main() -> None:
    """CLI entry point: plot each requested element, one figure per element."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wfc', '-w', required=True, help='Path to wfc.hsd (for the exact STO target overlay)')
    ap.add_argument('--derived-dir', '-d', required=True, help='Directory of derived .basis files to plot')
    ap.add_argument('--reference-dir', '-r', default=None, help='Optional directory of reference .basis files to overlay')
    ap.add_argument('--elements', '-e', nargs='*', default=None, help='Elements to plot, e.g. C N O (default: every *.basis file in --derived-dir)')
    ap.add_argument('--rmax', '-x', type=float, default=8.0, help='Plot range in bohr')
    ap.add_argument('--npoints', '-n', type=int, default=2000)
    ap.add_argument('--outdir', '-o', default=None, help='Save one <elem>.png per element into this directory instead of opening interactive windows')
    args = ap.parse_args()

    if args.outdir:
        matplotlib.use('Agg')

    wfc = parse_wfc(args.wfc)
    elements = args.elements if args.elements else find_elements(args.derived_dir)
    if not elements:
        raise SystemExit(f'Nothing to plot: no .basis files found in {args.derived_dir}')

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for elem in elements:
        if elem not in wfc:
            raise KeyError(f'{elem} not present in {args.wfc}')
        fig = plot_element(wfc, args.derived_dir, args.reference_dir, elem, args.rmax, args.npoints)
        if outdir:
            path = outdir / f'{elem}.png'
            fig.savefig(path, dpi=130) # type: ignore
            plt.close(fig)
            print(f'Saved {path}')

    if not outdir:
        plt.show()


if __name__ == '__main__':
    main()
