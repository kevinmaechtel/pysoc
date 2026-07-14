#!/usr/bin/env python3
"""Fit contracted GTOs to DFTB+ STO valence orbitals for use with PySOC.

Reproduces the mio-1-1-fit .basis convention from Gao et al., J. Chem.
Theory Comput. 2017, 13, 515-524 (10.1021/acs.jctc.6b00915): STO valence
orbitals from wfc.hsd (bohr^-1 exponents) are fit with contracted,
unnormalized GTOs (r^l * exp(-a r^2), bohr^-2 exponents).

The paper's Supporting Information (SI-2) states fitting errors of
1e-5 to 1e-7 but gives no algorithm, software, or exact error metric, and
that level is not reproduced by the shipped mio-1-1-fit/*.basis files
themselves when checked against the exact analytic wfc.hsd target (see
target_radial_from_wfc) with a standard r^2-weighted relative RMS - those
files sit at ~1e-3, matching what this script achieves. STO-to-GTO fitting
is a non-unique nonlinear least-squares problem: many different exponent/
coefficient sets reproduce the same radial function equally well, so an
independently-run fit is not expected to reproduce a reference .basis file
coefficient-for-coefficient. Use compare_basis_curves (which compares the
actual contracted radial functions) rather than raw coefficient diffs to
judge whether a derived basis matches a reference one.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

L2CHAR = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}
CHAR2L = {v: k for k, v in L2CHAR.items()}

DEFAULT_RMAX = 12.0
DEFAULT_NGRID = 2000
DEFAULT_WEIGHT = 'r2'
DEFAULT_NSTARTS = 60
DEFAULT_SEED = 0


def require_search(
        pattern: str,
        text: str,
        flags: int = 0
    ) -> re.Match[str]:
    """Like re.search, but raise ValueError instead of returning None."""
    match = re.search(pattern, text, flags)
    if match is None:
        raise ValueError(f'Pattern not found: {pattern}')
    return match


def parse_blocks(
        text: str
    ) -> dict[str, str]:
    """Split a HSD-style file into top-level ``Elem { ... }`` blocks.

    HSD allows the '=' before a block's '{' to be omitted (both
    ``"Elem = { ... }"`` and ``"Elem { ... }"`` are valid and appear in
    the wild - e.g. mio-1-1/ob2-1-1 use the former, 3ob-3-1 the latter),
    so both are accepted here.

    Returns a dict mapping each top-level element symbol to its full
    block text, including the braces.
    """
    entries: dict[str, str] = {}
    i = 0
    n = len(text)
    while i < n:
        m = re.search(r'(?m)^\s*([A-Z][a-z]?)\s*(?:=\s*)?\{', text[i:])
        if not m:
            break
        start = i + m.start()
        elem = m.group(1)
        brace0 = i + m.end() - 1
        depth = 0
        j = brace0
        while j < n:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    block = text[start:j+1]
                    entries[elem] = block
                    i = j + 1
                    break
            j += 1
        else:
            raise ValueError(f'Unclosed block for {elem}')
    return entries


def extract_braced_block(
        text: str,
        open_brace_index: int
    ) -> str:
    """Return the balanced ``{ ... }`` substring starting at a '{' index."""
    depth = 0
    for j in range(open_brace_index, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[open_brace_index:j + 1]
    raise ValueError('Unclosed brace block')


def parse_wfc(
        path: str | Path
    ) -> dict[str, dict[str, Any]]:
    """Parse a wfc.hsd file into per-element STO orbital data.

    Returns a dict keyed by element symbol, each value
    ``{'atomic_number': int, 'orbitals': [orbital, ...]}`` where each
    orbital dict has keys 'l' (int), 'occupation' (float), 'cutoff'
    (float), 'sto_exponents' (ndarray, shape (n_exponents,)) and
    'sto_coeff_matrix' (ndarray, shape (n_exponents, n_poly_terms)).
    """
    text = Path(path).read_text()
    blocks = parse_blocks(text)
    data: dict[str, dict[str, Any]] = {}
    for elem, block in blocks.items():
        z = int(require_search(r'AtomicNumber\s*=\s*(\d+)', block).group(1))
        orbitals = []
        for om in re.finditer(r'(?m)^\s*Orbital\s*(?:=\s*)?\{', block):
            ob = extract_braced_block(block, om.end() - 1)
            shell_l = int(require_search(r'AngularMomentum\s*=\s*(\d+)', ob).group(1))
            occ = float(require_search(r'Occupation\s*=\s*([0-9Ee+\-.]+)', ob).group(1))
            cutoff = float(require_search(r'Cutoff\s*=\s*([0-9Ee+\-.]+)', ob).group(1))
            exptxt = require_search(r'Exponents\s*(?:=\s*)?\{(.*?)\}', ob, re.S).group(1)
            coeftxt = require_search(r'Coefficients\s*(?:=\s*)?\{(.*?)\}', ob, re.S).group(1)
            exponents = np.array([float(x) for x in exptxt.split()], dtype=float)
            coeffs = np.array([float(x) for x in coeftxt.split()], dtype=float)
            if coeffs.size % exponents.size != 0:
                raise ValueError(f'{elem} l={shell_l}: coefficient count incompatible with exponent count')
            # Coefficients are stored exponent-major with polynomial terms per exponent.
            # Use shape (n_exponents, n_poly_terms) so each row corresponds to one exponent.
            coeffs = coeffs.reshape((exponents.size, -1))
            orbitals.append({
                'l': shell_l,
                'occupation': occ,
                'cutoff': cutoff,
                'sto_exponents': exponents,
                'sto_coeff_matrix': coeffs,
            })
        data[elem] = {'atomic_number': z, 'orbitals': orbitals}
    return data


def parse_skdef_valence(
        path: str | Path
    ) -> dict[str, list[str]]:
    """Read the ``ValenceShells`` list per element from a skdef.hsd file.

    Returns a dict mapping element symbol to a list of shell labels like
    ``['2s', '2p']``. Elements with no AtomParameters block yield {}.
    """
    text = Path(path).read_text()
    m = re.search(r'(?m)^\s*AtomParameters\s*\{', text)
    if not m:
        return {}
    body = extract_braced_block(text, m.end() - 1)
    out: dict[str, list[str]] = {}
    for om in re.finditer(r'(?m)^\s*([A-Z][a-z]?)\s*\{', body):
        elem = om.group(1)
        block = extract_braced_block(body, om.end() - 1)
        m_shells = re.search(r'ValenceShells\s*=\s*([^\n]+)', block)
        shells = m_shells.group(1).split() if m_shells else []
        out[elem] = shells
    return out


def parse_compression_spec(
        spec_text: str
    ) -> dict[str, Any]:
    """Parse a compression spec like ``PowerCompression { Power = 2; Radius = 7.0 }``.

    Returns ``{'type': str, 'parameters': dict[str, float | int]}``.
    """
    header = spec_text.strip()
    match = re.match(r'^([A-Za-z0-9_-]+)\s*(?:\{(.*)\})?$', header, re.S)
    if not match:
        return {'type': header, 'parameters': {}}

    compression_type = match.group(1)
    param_text = (match.group(2) or '').strip()
    parameters: dict[str, Any] = {}
    for token in re.split(r'[;\n]+', param_text):
        token = token.strip()
        if not token:
            continue
        kv = re.match(r'([A-Za-z0-9_-]+)\s*=\s*([0-9Ee+\-.]+)', token)
        if kv:
            value_text = kv.group(2)
            value = float(value_text) if any(ch in value_text for ch in '.eE') else int(value_text)
            parameters[kv.group(1)] = value
        else:
            parameters.setdefault('raw', []).append(token)

    return {'type': compression_type, 'parameters': parameters}


def parse_skdef_compressions(
        path: str | Path
    ) -> dict[str, dict[str, Any]]:
    """Read DensityCompression / WaveCompressions per element from skdef.hsd.

    Returns a dict mapping element symbol to
    ``{'density_compression': spec, 'wave_compressions': {shell_label: spec}}``,
    where each spec is the dict returned by parse_compression_spec.
    """
    text = Path(path).read_text()
    m = re.search(r'(?m)^\s*AtomParameters\s*\{', text)
    if not m:
        return {}

    body = extract_braced_block(text, m.end() - 1)
    out: dict[str, dict[str, Any]] = {}
    for om in re.finditer(r'(?m)^\s*([A-Z][a-z]?)\s*\{', body):
        elem = om.group(1)
        block = extract_braced_block(body, om.end() - 1)
        dftb_match = re.search(r'(?m)^\s*DftbAtom\s*\{', block)
        if not dftb_match:
            continue

        dftb_block = extract_braced_block(block, dftb_match.end() - 1)
        density_match = re.search(r'DensityCompression\s*=\s*([^\n\{]+(?:\{.*?\})?)', dftb_block, re.S)
        wave_match = re.search(r'(?m)^\s*WaveCompressions\s*=\s*SingleAtomCompressions\s*\{', dftb_block)

        compression: dict[str, Any] = {}
        if density_match:
            compression['density_compression'] = parse_compression_spec(density_match.group(1))

        if wave_match:
            wave_block = extract_braced_block(dftb_block, wave_match.end() - 1)
            wave_compressions = {}
            for sm in re.finditer(r'(?m)^\s*([A-Za-z0-9_-]+)\s*=\s*([^\n\{]+(?:\{.*?\})?)', wave_block, re.S):
                shell_label = sm.group(1)
                wave_compressions[shell_label] = parse_compression_spec(sm.group(2))
            compression['wave_compressions'] = wave_compressions

        if compression:
            out[elem] = compression

    return out


def compression_radius(
        compression_spec: dict[str, Any] | None,
        default: float | None = None
    ) -> float | None:
    """Extract the ``Radius`` parameter from a parse_compression_spec dict."""
    if not compression_spec:
        return default
    params = compression_spec.get('parameters', {})
    radius = params.get('Radius')
    if radius is None:
        return default
    return float(radius)


def shell_fit_rmax(
        elem_compression: dict[str, Any] | None,
        shell_label: str,
        default_rmax: float
    ) -> float:
    """Pick a fit cutoff radius for one shell from its skdef compression info.

    Prefers the shell's own WaveCompression radius, falls back to the
    element's DensityCompression radius, then to default_rmax.
    """
    if not elem_compression:
        return default_rmax

    wave = elem_compression.get('wave_compressions', {})
    if shell_label in wave:
        return compression_radius(wave[shell_label], default_rmax) # type: ignore

    density = elem_compression.get('density_compression')
    return compression_radius(density, default_rmax) # type: ignore


def sto_primitive_radial(
        r: np.ndarray,
        zeta: float, 
        power: int
    ) -> np.ndarray:
    """Evaluate one unnormalized Slater-type primitive r^power * exp(-zeta r)."""
    return (r**power) * np.exp(-zeta * r)


def target_radial_from_wfc(
        orb: dict[str, Any]
    ) -> Callable[[np.ndarray], np.ndarray]:
    """Build R(r) for one wfc.hsd orbital as an exact analytic function.

    R(r) = sum_zeta sum_k c[zeta,k] * r^(l+k) * exp(-zeta r), using every
    polynomial term for every exponent. Verified against the raw
    slateratom coefficient files (coeffs_0*.tag in a skprogs _build/
    comp.* directory matching the element's WaveCompressions radius) to
    ~3e-7: this is the exact analytic form of the wave-compressed orbital
    skgen writes to wfc.hsd.

    Returns a callable r -> R(r) (r may be a scalar-shaped ndarray).
    """
    shell_l = orb['l']
    zetas = orb['sto_exponents']
    coeff_matrix = orb['sto_coeff_matrix']

    def f(r: np.ndarray) -> np.ndarray:
        vals = np.zeros_like(r)
        for zeta, coeffs_for_zeta in zip(zetas, coeff_matrix):
            for k, c in enumerate(coeffs_for_zeta):
                vals += c * sto_primitive_radial(r, zeta, shell_l + k)
        return vals
    return f


def parse_atom_wave_file(
        path: str | Path
    ) -> dict[str, Any]:
    """Parse a skprogs slateratom ``wave_*.dat`` numerical wavefunction file.

    Returns a dict with 'path', 'principal_qn', 'l', 'occupation', 'r',
    'wave' (all ndarrays for the array fields), plus 'wave_1st'/'wave_2nd'
    when those derivative columns are present in the file.
    """
    lines = Path(path).read_text().splitlines()
    principal_qn = None
    shell_l = None
    occupation = None
    data_start = None

    for idx, line in enumerate(lines):
        m = re.search(r'Principal QN=\s*(\d+)\s*,\s*l=\s*(\d+)\s*,\s*Occupation=\s*([0-9Ee+\-.]+)', line)
        if m:
            principal_qn = int(m.group(1))
            shell_l = int(m.group(2))
            occupation = float(m.group(3))
            data_start = idx + 1
            break

    if data_start is None:
        raise ValueError(f'Could not locate wavefunction header in {path}')

    r = []
    wave = []
    wave_1st = []
    wave_2nd = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        r.append(float(parts[0]))
        wave.append(float(parts[2]))
        if len(parts) > 3:
            wave_1st.append(float(parts[3]))
        if len(parts) > 4:
            wave_2nd.append(float(parts[4]))

    out: dict[str, Any] = {
        'path': str(path),
        'principal_qn': principal_qn,
        'l': shell_l,
        'occupation': occupation,
        'r': np.array(r, dtype=float),
        'wave': np.array(wave, dtype=float),
    }

    if wave_1st:
        out['wave_1st'] = np.array(wave_1st, dtype=float)
    if wave_2nd:
        out['wave_2nd'] = np.array(wave_2nd, dtype=float)

    return out


def find_atom_wave_directory(
        atom_root: str | Path | None,
        elem: str
    ) -> Path | None:
    """Locate the first skprogs ``atom.*/atom0`` output directory for elem."""
    if not atom_root:
        return None

    elem_dir = Path(atom_root) / elem.lower()
    if not elem_dir.exists():
        return None

    candidates = sorted(elem_dir.glob('atom.*/atom0'))
    return candidates[0] if candidates else None


def load_atom_wave_target(
        atom_root: str | Path | None,
        elem: str, 
        shell_label: str
    ) -> dict[str, Any] | None:
    """Load the numerical wavefunction for one element/shell from _build output.

    Prefers a spin-down, then spin-up, then any matching wave_*.dat file
    under the element's atom0 directory; among files for the requested
    angular momentum, picks the one with the highest principal quantum
    number. Returns None if atom_root is unset or nothing matches.
    """
    atom_dir = find_atom_wave_directory(atom_root, elem)
    if atom_dir is None:
        return None

    shell_key = shell_label.lower()
    preferred = []
    preferred.extend(sorted(atom_dir.glob(f'wave_*{shell_key}_dn.dat')))
    preferred.extend(sorted(atom_dir.glob(f'wave_*{shell_key}_up.dat')))
    preferred.extend(sorted(atom_dir.glob(f'wave_*{shell_key}*.dat')))
    if not preferred:
        return None

    parsed = [parse_atom_wave_file(path) for path in preferred]
    parsed = [item for item in parsed if item['l'] == CHAR2L[shell_label]]
    if not parsed:
        return None

    parsed.sort(key=lambda item: (item['principal_qn'] is None, item['principal_qn'] or -1, item['path']))
    return parsed[-1]


def gto_contract_radial(
        r: np.ndarray,
        alphas: np.ndarray, 
        coeffs: np.ndarray, 
        shell_l: int
    ) -> np.ndarray:
    """Evaluate a contracted GTO shell R(r) = sum_i c_i * r^l * exp(-a_i r^2).

    Primitives are unnormalized r^l * exp(-a r^2): this matches the PySOC
    mio-1-1-fit .basis convention, verified by direct comparison against
    the reference C.basis / N.basis files (normalized primitives do not
    reproduce those files; raw primitives do, to ~1e-3 relative rms).
    """
    vals = np.zeros_like(r)
    for a, c in zip(alphas, coeffs):
        vals += c * (r**shell_l) * np.exp(-a * r*r)
    return vals


def make_grid(
        rmax: float = DEFAULT_RMAX, 
        ngrid: int = DEFAULT_NGRID
    ) -> np.ndarray:
    """Build a linearly-spaced radial grid on (0, rmax] with ngrid points."""
    return np.linspace(1e-6, rmax, ngrid)


def weights(
        r: np.ndarray,
        scheme: str = DEFAULT_WEIGHT
    ) -> np.ndarray:
    """Fit-residual weight profile over r for one of 'r2', 'r0', 'soc'.

    'r2' matches the physical 3D radial normalization weight r^2 (the
    default and the one used for all production fits); 'r0' is uniform;
    'soc' is ~1/r, emphasizing the near-nucleus region relevant to the
    Breit-Pauli spin-orbit operator (see fit_shell_varpro's docstring for
    why this is not the default).
    """
    if scheme == 'r2':
        return r*r
    if scheme == 'r0':
        return np.ones_like(r)
    if scheme == 'soc':
        return np.where(r > 1e-4, 1.0 / np.maximum(r, 1e-4), 1e4)
    raise ValueError(f'Unknown weight scheme {scheme}')


def compression_taper(
        r: np.ndarray, 
        rmax: float | None
    ) -> np.ndarray:
    """Smooth exp(-(r/rmax)^4) taper used to soft-cutoff the fit weight at rmax."""
    if rmax is None or rmax <= 0:
        return np.ones_like(r)
    scaled = r / float(rmax)
    return np.exp(-(scaled ** 4))


def primitive_matrix(
        r: np.ndarray,
        alphas: np.ndarray, 
        shell_l: int
    ) -> np.ndarray:
    """Design matrix of unnormalized primitives: column i is r^l * exp(-a_i r^2)."""
    return np.vstack([(r**shell_l) * np.exp(-a * r * r) for a in alphas]).T


def solve_linear_coeffs(
        A: np.ndarray, 
        w: np.ndarray, 
        target: np.ndarray, 
        ridge: float = 1e-12
    ) -> np.ndarray:
    """Weighted least-squares contraction coefficients for fixed exponents.

    Solves min_c ||w*(A c - target)||^2 with a light relative ridge
    penalty (scaled to the design matrix's own magnitude) that only bites
    when two exponents become nearly collinear. Without it, VARPRO
    occasionally drives a near-duplicate exponent pair to a huge,
    numerically meaningless value (e.g. alpha~1e17) with a canceling
    coefficient pair, since that direction is almost flat in the
    objective. ridge=1e-12 eliminates that failure mode without
    measurably changing well-conditioned fits.
    """
    wA = w[:, None] * A
    wt = w * target
    scale = np.mean(np.sum(wA ** 2, axis=0))
    lam = ridge * scale
    wA = np.vstack([wA, np.sqrt(lam) * np.eye(A.shape[1])])
    wt = np.concatenate([wt, np.zeros(A.shape[1])])
    coeffs, *_ = np.linalg.lstsq(wA, wt, rcond=None)
    return coeffs


def fit_shell_varpro(
    r: np.ndarray,
    target: np.ndarray,
    shell_l: int,
    nprim: int,
    amin: float = 0.05,
    amax: float = 3000.0,
    weight_scheme: str = 'r2',
    nstarts: int = DEFAULT_NSTARTS,
    seed: int = DEFAULT_SEED,
    fit_rmax: float | None = None,
    use_compression_weight: bool = False,
) -> dict[str, Any]:
    """Fit a contracted GTO shell to a target radial function.

    Uses variable projection (Golub-Pereyra): only the nprim exponents
    are optimized nonlinearly; for any trial set of exponents the
    contraction coefficients are the exact linear-least-squares solution
    (solve_linear_coeffs). This halves the parameter count versus jointly
    optimizing exponents and coefficients, converges faster and to a
    lower residual, and avoids the spurious near-degenerate primitives
    that joint optimization produced for wide-exponent-range shells.

    Tries `nstarts` randomized initial exponent guesses (log-normal
    perturbations of a geomspace(amin, amax, nprim) baseline) and keeps
    the best (lowest weighted relative RMS) result.

    Returns a dict with keys 'alphas', 'coeffs' (ndarrays, best fit),
    'fit' (ndarray, the fitted curve on `r`), 'rel_rms' (float),
    'success' (bool, from the winning least_squares call), 'message',
    'cost', 'nfev', and 'attempts' (1-indexed attempt that won).
    """
    w = np.sqrt(weights(r, weight_scheme))
    if use_compression_weight:
        w = w * np.sqrt(compression_taper(r, fit_rmax))
    weight_vector = w * w

    base = np.geomspace(amin, amax, nprim)
    rng = np.random.default_rng(seed)

    # Unbounded log-alpha steps can transiently overflow exp() during line
    # search before being rejected; harmless, so silence the warning.
    np.seterr(over='ignore')

    def resid(log_alpha: np.ndarray) -> np.ndarray:
        A = primitive_matrix(r, np.exp(log_alpha), shell_l)
        coeffs = solve_linear_coeffs(A, w, target)
        return w * (A @ coeffs - target)

    best: dict[str, Any] | None = None
    for attempt in range(max(1, int(nstarts))):
        if attempt == 0:
            alpha0 = base
        else:
            spread = rng.normal(scale=1.2, size=nprim)
            alpha0 = np.clip(np.exp(np.log(base) + spread), amin * 1e-3, amax * 1e3)

        res = least_squares(resid, np.log(alpha0), max_nfev=3000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        alphas = np.exp(res.x)
        A = primitive_matrix(r, alphas, shell_l)
        coeffs = solve_linear_coeffs(A, w, target)
        fit = A @ coeffs
        rel_rms = float(np.sqrt(
            np.sum(weight_vector * (fit - target) ** 2) / np.sum(weight_vector * target ** 2)
        ))

        if best is None or rel_rms < best['rel_rms']:
            best = {
                'alphas': alphas,
                'coeffs': coeffs,
                'fit': fit,
                'rel_rms': rel_rms,
                'success': bool(res.success),
                'message': res.message,
                'cost': float(res.cost),
                'nfev': int(res.nfev),
                'attempts': attempt + 1,
            }

    assert best is not None  # loop runs at least once (max(1, nstarts))
    return best


def choose_valence_orbital(
        orbitals_by_l: dict[int, list[dict[str, Any]]],
        preferred: str = 'last'
    ) -> dict[int, dict[str, Any]]:
    """Pick one orbital per angular momentum when wfc.hsd lists several.

    `preferred` is 'last', 'first', or a string integer index into each
    l's orbital list.
    """
    chosen = {}
    for shell_l, arr in orbitals_by_l.items():
        if preferred == 'last':
            chosen[shell_l] = arr[-1]
        elif preferred == 'first':
            chosen[shell_l] = arr[0]
        else:
            idx = int(preferred)
            chosen[shell_l] = arr[idx]
    return chosen


def format_basis(
        elem: str, 
        shells: dict[int, dict[str, Any]]
    ) -> str:
    """Render fitted shells (as returned by fit_shell_varpro) as .basis text."""
    lines = [f'{elem} 0']
    for shell_l in sorted(shells):
        label = L2CHAR[shell_l]
        alphas = shells[shell_l]['alphas']
        coeffs = shells[shell_l]['coeffs']
        order = np.argsort(alphas)
        lines.append(f'{label} {len(alphas)} 1.0')
        for a, c in zip(alphas[order], coeffs[order]):
            lines.append(f'{a: .10E} {c: .10E}')
    return '\n'.join(lines) + '\n'


def extract_orbitals_by_l(
        elem_entry: dict[str, Any]
    ) -> dict[int, list[dict[str, Any]]]:
    """Group a parse_wfc element entry's orbitals by angular momentum."""
    out: dict[int, list[dict[str, Any]]] = {}
    for orb in elem_entry['orbitals']:
        out.setdefault(orb['l'], []).append(orb)
    return out


def parse_basis_file(
        path: str | Path
    ) -> tuple[str, dict[int, dict[str, Any]]]:
    """Parse a PySOC-style .basis file.

    Returns (element_symbol, shells) where shells maps angular momentum
    (int) to {'label': str, 'exponents': ndarray, 'coeffs': ndarray}.
    """
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f'Empty basis file: {path}')

    element = lines[0].split()[0]
    shells: dict[int, dict[str, Any]] = {}
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        shell_label = parts[0].upper()
        if shell_label not in CHAR2L:
            raise ValueError(f'Unexpected shell label {shell_label} in {path}')
        nprim = int(parts[1])
        i += 1
        exponents = []
        coeffs = []
        for _ in range(nprim):
            exp_str, coeff_str = lines[i].split()[:2]
            exponents.append(float(exp_str))
            coeffs.append(float(coeff_str))
            i += 1
        shell_l = CHAR2L[shell_label]
        shells[shell_l] = {
            'label': shell_label,
            'exponents': np.array(exponents, dtype=float),
            'coeffs': np.array(coeffs, dtype=float),
        }
    return element, shells


def reference_nprim_for_element(
        reference_dir: str | Path, 
        elem: str
    ) -> dict[int, int]:
    """Read the exact per-shell primitive count for elem from a reference .basis dir.

    The mio-1-1-fit primitive counts are not uniform across elements
    (e.g. H uses 7 S-primitives, O uses 9) - they look like they were
    chosen per shell, not fixed globally. Read them from a reference
    .basis file instead of assuming a fixed count, when one is available.
    Returns {} if reference_dir has no <elem>.basis file.
    """
    path = Path(reference_dir) / f'{elem}.basis'
    if not path.exists():
        return {}
    _, shells = parse_basis_file(path)
    return {shell_l: sh['exponents'].size for shell_l, sh in shells.items()}


def compare_basis_files(
        reference_path: str | Path, 
        derived_path: str | Path
    ) -> dict[str, Any]:
    """Diff raw exponents/coefficients between two .basis files, shell by shell.

    Primitives are matched by sorting each shell's exponents ascending.
    Note: because STO-to-GTO fitting is non-unique (see module
    docstring), a large diff here does not imply a bad fit - use
    compare_basis_curves for that judgement instead.
    """
    ref_element, ref_shells = parse_basis_file(reference_path)
    derived_element, derived_shells = parse_basis_file(derived_path)
    if ref_element != derived_element:
        raise ValueError(f'Element mismatch: {ref_element} vs {derived_element}')

    shell_report: dict[str, Any] = {}
    all_shells = sorted(set(ref_shells) | set(derived_shells))
    for shell_l in all_shells:
        if shell_l not in ref_shells:
            shell_report[L2CHAR[shell_l]] = {'status': 'missing_in_reference'}
            continue
        if shell_l not in derived_shells:
            shell_report[L2CHAR[shell_l]] = {'status': 'missing_in_derived'}
            continue

        ref = ref_shells[shell_l]
        derived = derived_shells[shell_l]
        ref_order = np.argsort(ref['exponents'])
        derived_order = np.argsort(derived['exponents'])
        ref_exp = ref['exponents'][ref_order]
        derived_exp = derived['exponents'][derived_order]
        ref_coeff = ref['coeffs'][ref_order]
        derived_coeff = derived['coeffs'][derived_order]

        n = min(ref_exp.size, derived_exp.size)
        exp_diff = derived_exp[:n] - ref_exp[:n]
        coeff_diff = derived_coeff[:n] - ref_coeff[:n]
        shell_report[L2CHAR[shell_l]] = {
            'status': 'compared',
            'reference_nprim': int(ref_exp.size),
            'derived_nprim': int(derived_exp.size),
            'matched_primitives': int(n),
            'exponent_max_abs_diff': float(np.max(np.abs(exp_diff))) if n else None,
            'exponent_rms_diff': float(np.sqrt(np.mean(exp_diff**2))) if n else None,
            'coeff_max_abs_diff': float(np.max(np.abs(coeff_diff))) if n else None,
            'coeff_rms_diff': float(np.sqrt(np.mean(coeff_diff**2))) if n else None,
            'exponent_allclose': bool(ref_exp.size == derived_exp.size and np.allclose(ref_exp, derived_exp)),
            'coeff_allclose': bool(ref_coeff.size == derived_coeff.size and np.allclose(ref_coeff, derived_coeff)),
        }

    return {
        'reference_path': str(reference_path),
        'derived_path': str(derived_path),
        'element': ref_element,
        'shells': shell_report,
    }


def compare_basis_curves(
    reference_path: str | Path,
    derived_path: str | Path,
    rmax: float = DEFAULT_RMAX,
    ngrid: int = DEFAULT_NGRID,
) -> dict[str, Any]:
    """Compare two .basis files by their contracted radial curves, not raw numbers.

    Nonlinear GTO fits are non-unique: two parameter sets can represent
    essentially the same radial function while differing coefficient-by-
    coefficient. Compare the actual contracted radial curves instead.
    This is the metric to use to judge whether a derived basis matches a
    reference one.
    """
    ref_element, ref_shells = parse_basis_file(reference_path)
    derived_element, derived_shells = parse_basis_file(derived_path)
    if ref_element != derived_element:
        raise ValueError(f'Element mismatch: {ref_element} vs {derived_element}')

    r = make_grid(rmax, ngrid)
    shell_report: dict[str, Any] = {}
    for shell_l in sorted(set(ref_shells) | set(derived_shells)):
        if shell_l not in ref_shells or shell_l not in derived_shells:
            shell_report[L2CHAR[shell_l]] = {'status': 'missing'}
            continue
        ref = ref_shells[shell_l]
        derived = derived_shells[shell_l]
        ref_curve = gto_contract_radial(r, ref['exponents'], ref['coeffs'], shell_l)
        derived_curve = gto_contract_radial(r, derived['exponents'], derived['coeffs'], shell_l)
        rel_rms = float(np.sqrt(
            np.sum(r**2 * (derived_curve - ref_curve) ** 2) / np.sum(r**2 * ref_curve ** 2)
        ))
        shell_report[L2CHAR[shell_l]] = {
            'status': 'compared',
            'relative_rms_curve_diff': rel_rms,
            'max_abs_curve_diff': float(np.max(np.abs(derived_curve - ref_curve))),
        }
    return {
        'reference_path': str(reference_path),
        'derived_path': str(derived_path),
        'element': ref_element,
        'shells': shell_report,
    }


def comparison_exceeds_thresholds(
    comparison_report: dict[str, Any],
    exponent_threshold: float | None = None,
    coeff_threshold: float | None = None,
) -> list[str]:
    """Return the shell labels in a compare_basis_files report exceeding thresholds.

    A shell also counts as exceeded if it wasn't 'compared' (e.g. missing
    from one side). Either threshold may be omitted to skip that check.
    """
    exceeded = []
    for shell_label, shell_report in comparison_report.get('shells', {}).items():
        if shell_report.get('status') != 'compared':
            exceeded.append(shell_label)
            continue
        exp_diff = shell_report.get('exponent_max_abs_diff')
        coeff_diff = shell_report.get('coeff_max_abs_diff')
        if exponent_threshold is not None and exp_diff is not None and exp_diff > exponent_threshold:
            exceeded.append(shell_label)
        if coeff_threshold is not None and coeff_diff is not None and coeff_diff > coeff_threshold:
            exceeded.append(shell_label)
    return sorted(set(exceeded))


def select_chosen_shells(
    wfc: dict[str, dict[str, Any]],
    valence: dict[str, list[str]],
    elem: str,
    prefer_orbital: str,
) -> dict[int, dict[str, Any]]:
    """Pick the orbital to fit per angular momentum for elem, filtered to valence shells.

    Falls back to fitting every orbital present in wfc.hsd for elem when
    `valence` has no entry for it (e.g. no --skdef was given).
    """
    orbitals_by_l = extract_orbitals_by_l(wfc[elem])
    chosen = choose_valence_orbital(orbitals_by_l, preferred=prefer_orbital)
    valence_shell_labels = valence.get(elem, [])
    if valence_shell_labels:
        valence_ls = {CHAR2L[label[-1].upper()] for label in valence_shell_labels if label[-1].upper() in CHAR2L}
        chosen = {shell_l: orb for shell_l, orb in chosen.items() if shell_l in valence_ls}
    return chosen


def fit_shell_target(
    orb: dict[str, Any],
    shell_l: int,
    elem: str,
    fit_rmax: float,
    atom_root: str | Path | None,
    use_atom_wave_target: bool,
    ngrid: int,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Build the (r, target) fit target for one shell.

    Uses the numerical _build wavefunction under atom_root when
    use_atom_wave_target is set and one is found, otherwise the exact
    analytic wfc.hsd STO expansion on a fresh grid up to fit_rmax.

    Returns (r, target, atom_wave_path) where atom_wave_path is the
    source file path when the numerical target was used, else None.
    """
    shell_label = L2CHAR[shell_l]
    if use_atom_wave_target and atom_root:
        atom_target = load_atom_wave_target(atom_root, elem, shell_label)
        if atom_target is None:
            raise FileNotFoundError(f'No atom wave target found for {elem} shell {shell_label} under {atom_root}')
        return atom_target['r'], atom_target['wave'], atom_target['path']

    r_shell = make_grid(fit_rmax, ngrid)
    target = target_radial_from_wfc(orb)(r_shell)
    return r_shell, target, None


def fit_element(
    elem: str,
    wfc: dict[str, dict[str, Any]],
    valence: dict[str, list[str]],
    compressions: dict[str, dict[str, Any]],
    nprim_map: dict[int, int],
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Fit every valence shell of one element.

    Returns (fitted, meta): `fitted` maps angular momentum to the
    fit_shell_varpro result dict (consumed by format_basis); `meta` is
    the JSON-serializable per-element report entry.
    """
    chosen = select_chosen_shells(wfc, valence, elem, args.prefer_orbital)
    nprim_overrides = reference_nprim_for_element(args.nprim_from, elem) if args.nprim_from else {}

    fitted: dict[int, dict[str, Any]] = {}
    meta: dict[str, Any] = {
        'atomic_number': wfc[elem]['atomic_number'],
        'valence_shells': valence.get(elem, []),
        'fit_basis': True,
    }
    if elem in compressions:
        meta['compression'] = compressions[elem]

    for shell_l, orb in chosen.items():
        shell_label = L2CHAR[shell_l]
        fit_rmax = args.rmax
        if args.use_compression_rmax:
            fit_rmax = shell_fit_rmax(compressions.get(elem), shell_label, args.rmax)

        r_shell, target, atom_wave_path = fit_shell_target(
            orb, shell_l, elem, fit_rmax, args.atom_root, args.use_atom_wave_target, args.ngrid
        )
        if atom_wave_path:
            meta.setdefault('atom_wave_targets', {})[shell_label] = atom_wave_path

        nprim = nprim_overrides.get(shell_l, nprim_map[shell_l])
        fitres = fit_shell_varpro(
            r_shell,
            target,
            shell_l,
            nprim=nprim,
            weight_scheme=args.weight,
            nstarts=args.nstarts,
            seed=args.seed + shell_l,
            fit_rmax=fit_rmax,
            use_compression_weight=args.use_compression_weight,
        )
        fitted[shell_l] = fitres
        meta[shell_label] = {
            'occupation': orb['occupation'],
            'cutoff': orb['cutoff'],
            'sto_exponents': orb['sto_exponents'].tolist(),
            'sto_coeff_matrix': orb['sto_coeff_matrix'].tolist(),
            'relative_rms': fitres['rel_rms'],
            'success': fitres['success'],
            'message': fitres['message'],
            'fit_attempts': fitres['attempts'],
            'fit_rmax': fit_rmax,
            'compression_weight': bool(args.use_compression_weight),
        }
    return fitted, meta


def write_basis_and_compare(
    elem: str,
    fitted: dict[int, dict[str, Any]],
    meta: dict[str, Any],
    outdir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Write <outdir>/<elem>.basis and, if --compare-with is set, compare it.

    Adds 'comparison' (raw coefficient diff) and 'comparison_curves'
    (radial-function diff) entries to `meta`. Raises SystemExit if any
    shell exceeds --compare-exponent-threshold / --compare-coeff-threshold.
    Returns the (possibly updated) meta dict.
    """
    basis_txt = format_basis(elem, fitted)
    basis_path = outdir / f'{elem}.basis'
    basis_path.write_text(basis_txt)

    if not args.compare_with:
        return meta

    comparison = compare_basis_files(args.compare_with, basis_path)
    meta['comparison'] = comparison
    meta['comparison_curves'] = compare_basis_curves(args.compare_with, basis_path, rmax=args.rmax, ngrid=args.ngrid)

    exceeded_shells = comparison_exceeds_thresholds(
        comparison,
        exponent_threshold=args.compare_exponent_threshold,
        coeff_threshold=args.compare_coeff_threshold,
    )
    if exceeded_shells:
        comparison['status'] = 'failed'
        comparison['failed_shells'] = exceeded_shells
        raise SystemExit(f'Comparison failed for {elem}: exceeded thresholds in shells {", ".join(exceeded_shells)}')
    return meta


def main() -> None:
    """CLI entry point: fit .basis files for the requested elements and report."""
    ap = argparse.ArgumentParser(description='Derive PySOC-style .basis files from wfc.hsd STO orbitals.')
    ap.add_argument('--wfc', '-w', required=True, help='Path to wfc.hsd')
    ap.add_argument('--skdef', '-s', default=None, help='Optional skdef.hsd for valence-shell metadata')
    ap.add_argument('--elements', '-e', nargs='*', default=None, help='Elements to fit, e.g. C N O')
    ap.add_argument('--outdir', '-o', default='derived_basis')
    ap.add_argument('--weight', '-W', default='r2', choices=['r0', 'r2', 'soc'])
    ap.add_argument('--rmax', '-R', type=float, default=DEFAULT_RMAX)
    ap.add_argument('--ngrid', '-G', type=int, default=DEFAULT_NGRID)
    ap.add_argument('--prefer-orbital', '-P', default='last', help='Which STO column per l channel to fit: last, first, or integer index')
    ap.add_argument('--nprim-s', '-S', type=int, default=8)
    ap.add_argument('--nprim-p', '-p', type=int, default=7)
    ap.add_argument('--nprim-d', '-D', type=int, default=7)
    ap.add_argument('--nprim-f', '-F', type=int, default=7)
    ap.add_argument('--nprim-from', '-f', default='mio-1-1-fit', help='Directory of reference .basis files to read the exact per-shell primitive count from (per element, overriding --nprim-*); reference counts are not uniform (e.g. H uses 7 S-primitives, O uses 9). Falls back to --nprim-* for elements/shells missing from this directory. Pass a nonexistent path to disable.')
    ap.add_argument('--nstarts', '-N', type=int, default=DEFAULT_NSTARTS, help='Number of randomized fit restarts per shell')
    ap.add_argument('--seed', '-E', type=int, default=DEFAULT_SEED, help='Seed for randomized restarts')
    ap.add_argument('--use-compression-rmax', action='store_true', help='Use the skdef wave-compression radius as the fit cutoff for each shell when available')
    ap.add_argument('--use-compression-weight', action='store_true', help='Taper the fit weights using the compression radius when available')
    ap.add_argument('--atom-root', '-A', default=None, help='Optional _build root containing atom.* wavefunction outputs to use as direct fit targets when available')
    ap.add_argument('--use-atom-wave-target', action='store_true', help='Use atom wavefunction outputs from --atom-root as the fit target instead of the STO expansion from wfc.hsd')
    ap.add_argument('--compare-with', '-C', default=None, help='Optional basis file to compare against, for example C.basis_original')
    ap.add_argument('--compare-exponent-threshold', '-T', type=float, default=None, help='Fail if any shell exceeds this maximum absolute exponent difference')
    ap.add_argument('--compare-coeff-threshold', '-t', type=float, default=None, help='Fail if any shell exceeds this maximum absolute coefficient difference')
    ap.add_argument('--report-json', '-J', default=None)
    args = ap.parse_args()

    wfc = parse_wfc(args.wfc)
    valence = parse_skdef_valence(args.skdef) if args.skdef else {}
    compressions = parse_skdef_compressions(args.skdef) if args.skdef else {}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nprim_map = {0: args.nprim_s, 1: args.nprim_p, 2: args.nprim_d, 3: args.nprim_f}

    elements = args.elements if args.elements else sorted(wfc)
    report: dict[str, Any] = {}

    for elem in elements:
        if elem not in wfc:
            raise KeyError(f'{elem} not present in wfc.hsd')

        fitted, meta = fit_element(elem, wfc, valence, compressions, nprim_map, args)
        try:
            meta = write_basis_and_compare(elem, fitted, meta, outdir, args)
        finally:
            report[elem] = meta
            if args.report_json:
                Path(args.report_json).write_text(json.dumps(report, indent=2))

    if not args.report_json:
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
