# DFTB+ STO → GTO basis fitting for PySOC

Two scripts:

- **`derive_basis_from_wfc.py`** — fits contracted Gaussian-type orbitals
  (GTOs) to the Slater-type orbitals (STOs) DFTB+ uses, producing
  `.basis` files in the format PySOC expects for spin-orbit coupling (SOC)
  calculations.
- **`visualize_basis.py`** — plots the resulting radial functions against
  a reference and/or the exact STO target, to judge fit quality visually.

## Background

DFTB+ parameter sets (e.g. `mio-1-1`) represent valence atomic orbitals as
Slater-type orbitals. [PySOC](https://github.com/gaox-qd/pysoc) (Gao et al.,
*J. Chem. Theory Comput.* 2017, 13, 515–524,
[10.1021/acs.jctc.6b00915](https://doi.org/10.1021/acs.jctc.6b00915)) needs
Gaussian-type orbitals instead, so each parameter set ships a fitted
`.basis` file per element (see `mio-1-1-fit/*.basis`, taken from PySOC's own
repository).

Two inputs, produced by `skgen`/`skprogs` alongside a parameter set, drive
the fit:

- **`wfc.hsd`** — the exact analytic STO expansion of each element's
  valence orbitals (wave-compressed, i.e. using the atomic-orbital
  compression radius from `skdef.hsd`). This is the fitting *target*.
- **`skdef.hsd`** — the DFTB+ Slater-Koster generation input. Used here for
  `ValenceShells` (which angular momenta to actually fit) and
  `WaveCompressions` (optional per-shell fit cutoff radius).

`derive_basis_from_wfc.py` fits a `.basis` file for each element from these
two inputs alone. `mio-1-1-fit/` is also used as a *reference*: to validate
the fit (do we reproduce a known-good parameter set?) and to read the exact
number of Gaussian primitives per shell to use (see "Primitive counts"
below).

### Important: raw coefficients are not comparable between fits

STO→GTO fitting is a **non-unique nonlinear least-squares problem**: many
different exponent/coefficient combinations reproduce the same radial
function equally well (verified empirically — 300 random restarts on one
shell converge to visibly different-looking but equally-good solutions).
So an independently-run fit is **not expected** to reproduce a reference
`.basis` file coefficient-for-coefficient, even when it's a good fit. Two
files can differ completely number-by-number while representing the same
curve.

**Never compare `.basis` files by eyeballing the printed numbers.** Use:

- `derive_basis_from_wfc.compare_basis_curves()` (or `--compare-with` on
  the CLI, which calls it automatically) — compares the actual contracted
  radial *functions* on a grid. This is the number that matters.
- `visualize_basis.py` — plots the curves directly.

`compare_basis_files()` (raw coefficient/exponent diffing) also exists and
is used for exact-structure checks (e.g. that primitive counts match), but
a large diff there does not imply a bad fit.

Also note: the original paper's Supporting Information claims fitting
errors of 1e-5–1e-7, but this is not reproduced by the shipped
`mio-1-1-fit/*.basis` files themselves when checked against the exact
`wfc.hsd` target with a standard r²-weighted relative RMS — those files sit
at ~1e-3, matching what this script achieves. The SI number appears to
describe cherry-picked illustrative cases (its Figure 1S), not the general
fit quality. See the module docstring in `derive_basis_from_wfc.py` for
more detail.

---

## `derive_basis_from_wfc.py`

### How the fit works

For each requested element and each valence shell (angular momentum `l`):

1. Build the exact target radial function `R(r)` from `wfc.hsd`
   (`target_radial_from_wfc`) — a sum of Slater primitives
   `r^(l+k) * exp(-zeta*r)`.
2. Fit a contraction of `nprim` **unnormalized** Gaussian primitives,
   `R(r) ≈ Σ c_i · r^l · exp(-a_i r²)`, matching the raw-coefficient
   convention used by `mio-1-1-fit/*.basis` (confirmed: normalizing the
   primitives does *not* reproduce those files; raw primitives do).
3. Optimization uses **variable projection (VARPRO)**: only the `nprim`
   exponents are optimized nonlinearly (`fit_shell_varpro`); for any trial
   set of exponents, the contraction coefficients are the exact
   linear-least-squares solution (`solve_linear_coeffs`, with a tiny ridge
   penalty to prevent near-duplicate exponents from blowing up to
   numerically meaningless values). This is faster and more robust than
   jointly optimizing exponents and coefficients.
4. The fit is repeated from `--nstarts` randomized initial exponent
   guesses; the best (lowest weighted relative RMS) result is kept.
5. The residual is weighted by `r²` by default (`--weight r2`), matching
   the physical 3D radial normalization `∫|R(r)|² r² dr`. `r0` (uniform)
   and `soc` (~1/r, emphasizing the near-nucleus region relevant to the
   Breit-Pauli SOC operator) are also available but not default — they
   nail the near-origin value exactly at the cost of a much worse fit
   everywhere else.

### Primitive counts

Reference primitive counts are **not uniform** across elements — e.g.
`mio-1-1-fit/H.basis` uses 7 S-primitives, `O.basis` uses 9. By default
(`--nprim-from mio-1-1-fit`), the exact per-shell count is read from a
matching file in that reference directory when one exists, falling back to
`--nprim-s/-p/-d/-f` (default 8/7/7/7) for elements/shells missing from it
(as is the case for every `ob2-1-1` element). Pass a nonexistent path to
`--nprim-from` to disable and always use the `--nprim-*` values.

For elements with a much wider Slater-exponent range than mio-1-1's
lightest elements (e.g. `ob2-1-1`'s Zn/Br/I, ζ up to 53), 8/7/7 primitives
may not be enough for a good fit — pass a higher `--nprim-s/-p/-d` (10
worked well in practice; see the ridge-regularization note above for why
this is safe even when it creates near-collinear primitives).

### Usage

```bash
# Reproduce mio-1-1 (validates against the known-good reference)
python3 derive_basis_from_wfc.py \
    --wfc wfc.hsd --skdef skdef.hsd \
    --elements H C N O S P \
    --outdir mio-1-1-reproduced \
    --nstarts 60 --rmax 15 \
    --report-json mio-1-1-reproduced/report.json

# Fit a new parameter set with no reference available (ob2-1-1)
python3 derive_basis_from_wfc.py \
    --wfc ob2-1-1/wfc.hsd --skdef ob2-1-1/skdef.hsd \
    --elements H C N O F Mg P S Cl \
    --outdir ob2-1-1-fit --nstarts 60 --rmax 15

# Heavy elements needing more primitives (see "Primitive counts" above)
python3 derive_basis_from_wfc.py \
    --wfc ob2-1-1/wfc.hsd --skdef ob2-1-1/skdef.hsd \
    --elements Zn Br I \
    --outdir ob2-1-1-fit --nstarts 80 --rmax 15 \
    --nprim-s 10 --nprim-p 10 --nprim-d 10

# Validate one element against a reference, with a hard quality gate
python3 derive_basis_from_wfc.py \
    --wfc wfc.hsd --skdef skdef.hsd --elements C \
    --outdir /tmp/check --compare-with mio-1-1-fit/C.basis \
    --compare-exponent-threshold 1e-4
```

### Key CLI options

| Option | Short | Default | Meaning |
|---|---|---|---|
| `--wfc` | `-w` | *(required)* | Path to `wfc.hsd` |
| `--skdef` | `-s` | none | Path to `skdef.hsd`, for `ValenceShells` filtering and `WaveCompressions` cutoffs |
| `--elements` | `-e` | all in `wfc.hsd` | Elements to fit |
| `--outdir` | `-o` | `derived_basis` | Where to write `<elem>.basis` files |
| `--weight` | `-W` | `r2` | Fit weighting: `r2`, `r0`, or `soc` |
| `--rmax` | `-R` | 12.0 | Radial fit grid extent (bohr) |
| `--nstarts` | `-N` | 60 | Randomized fit restarts per shell (VARPRO attempts are cheap; more is better up to diminishing returns) |
| `--nprim-s`, `--nprim-p`, `--nprim-d`, `--nprim-f` | `-S`, `-p`, `-D`, `-F` | 8, 7, 7, 7 | Primitives per shell, used when `--nprim-from` has no match |
| `--nprim-from` | `-f` | `mio-1-1-fit` | Reference dir to read exact per-shell primitive counts from |
| `--use-compression-rmax` | | off | Use each shell's `skdef.hsd` `WaveCompressions` radius as its fit cutoff instead of `--rmax` |
| `--compare-with` | `-C` | none | Reference `.basis` file to diff/validate against (single path — all fitted elements are compared to it, so only useful for one element at a time) |
| `--compare-exponent-threshold`, `--compare-coeff-threshold` | `-T`, `-t` | none | Fail (`SystemExit`) if the raw coefficient/exponent diff exceeds this |
| `--report-json` | `-J` | stdout | Where to write the per-element fit report (occupation, cutoff, relative RMS, comparison results, etc.) |

Run `python3 derive_basis_from_wfc.py --help` for the full list (also
includes `--atom-root`/`--use-atom-wave-target` for fitting against
numerical `_build/` slateratom output instead of the analytic `wfc.hsd`
expansion, and `--use-compression-weight` to taper the fit weight beyond a
shell's compression radius).

---

## `visualize_basis.py`

Plots, per element, one subplot per shell showing the exact STO target
(from `wfc.hsd`), the reference curve (if `--reference-dir` is given), and
the derived curve — so you can see the *radial functions* match, rather
than comparing coefficients that are expected to differ (see above).

### Usage

```bash
# Derived vs reference vs exact target, one element, shown interactively
python3 visualize_basis.py -w wfc.hsd -d mio-1-1-reproduced -r mio-1-1-fit -e C

# Every element found in --derived-dir, no reference available
python3 visualize_basis.py -w ob2-1-1/wfc.hsd -d ob2-1-1-fit

# Save one PNG per element instead of opening windows
python3 visualize_basis.py -w wfc.hsd -d mio-1-1-reproduced -r mio-1-1-fit -o plots/
```

### CLI options

| Option | Short | Default | Meaning |
|---|---|---|---|
| `--wfc` | `-w` | *(required)* | Path to `wfc.hsd`, for the exact STO target overlay |
| `--derived-dir` | `-d` | *(required)* | Directory of derived `.basis` files to plot |
| `--reference-dir` | `-r` | none | Optional directory of reference `.basis` files to overlay |
| `--elements` | `-e` | every `.basis` in `--derived-dir` | Elements to plot |
| `--rmax` | `-x` | 8.0 | Plot range (bohr) |
| `--npoints` | `-n` | 2000 | Points per curve |
| `--outdir` | `-o` | none (interactive) | Save `<elem>.png` here instead of opening windows |

---

## Files in this directory

| Path | What it is |
|---|---|
| `wfc.hsd`, `skdef.hsd` | mio-1-1 STO/valence-shell input |
| `mio-1-1-fit/` | Reference `.basis` files (from PySOC), used to validate the fit and to read primitive counts |
| `mio-1-1-reproduced/` | Output of reproducing mio-1-1 with this script |
| `ob2-1-1/wfc.hsd`, `ob2-1-1/skdef.hsd` | ob2-1-1 STO/valence-shell input (no reference `.basis` files exist for this set) |
| `ob2-1-1-fit/` | Output of fitting ob2-1-1 with this script |
| `_build/` | Raw `skprogs`/`slateratom` output, used only to independently verify the `target_radial_from_wfc` reconstruction formula (not needed for normal use) |
