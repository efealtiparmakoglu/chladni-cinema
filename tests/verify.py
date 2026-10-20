#!/usr/bin/env python3
"""
Chladni sinema projesi — KAPILAR (gates).

    python3 tests/verify.py

Her kapı sayısal bir ölçüm yapar ve kanıtladığı şeyi yazar.
Blender YOK, yalnızca numpy + saf Python.

Fizik özeti (bkz. chladni.py):
  * Kare plaka üzerinde 2B dalga denklemi FDTD (leapfrog) ile çözülür:
        u_tt = c²∇²u - γ·u_t + f(x,y,t)
    Serbest (Neumann) kenar koşulu: ∂u/∂n = 0 -> plaka kenarları DÜĞÜM
    değil ANTİNOD'dur (gerçek Chladni plakası gibi); kum kenarlara
    birilmez, iç nodal çizgilere toplanır.
  * Merkezde sürekli harmonik sürücü (geniş Gauss "titrak").
  * Rezonans taraması: modal projeksiyonla ölçülen tepe frekansı, kare
    plakanın analitik mod frekansı f_mn = c·√(m²+n²)/(2L) ile karşılaştırılır.
  * Kum: tane dinamiği + klasik Chladni YAKALAMA ölçütü.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chladni as ch  # noqa: E402

# ------------------------------------------------------------------ ayarlar
GRID = 201
FREQ_MODE0 = ch.mode_frequency(2, 0)     # 1.41421 Hz  (2,0)+(0,2) -> halka
FREQ_MODE1 = ch.mode_frequency(2, 2)     # 2.00000 Hz  (2,2)         -> ızgara
SEED = 20240517
N_GRAINS = 2400
SETTLE = 30.0        # plakanın kararlı duruma gelmesi için simülasyon süresi
GRAIN_TIME = 22.0    # kumun toplanması için simülasyon süresi

_results = []


class Gate:
    def __init__(self, name, claim):
        self.name = name
        self.claim = claim

    def check(self, ok, detail):
        _results.append((self.name, self.claim, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {self.name}  {detail}", flush=True)
        return ok


# --------------------------------------------------------------- yardımcılar
def settled_plate(freq, n=GRID, settle=SETTLE, drive_amp=ch.DRIVE_AMP):
    """Verilen sürücü frekansında kararlı duruma ulaşmış plaka."""
    p = ch.Plate(n=n, damping=ch.DAMPING, drive_amp=drive_amp, freq=freq,
                 env_tau=1.5, boundary="neumann")
    p.run(settle)
    return p


def settled_grains(plate, seed=SEED, n_grains=N_GRAINS, t=GRAIN_TIME):
    """Plakanın kararlı zarfına göre toplanmış kum."""
    amp = plate.amplitude(normalize=True)
    gx, gy = ch._grad2(amp, plate.dx)
    g = ch.GrainField(n_grains=n_grains, seed=seed, L=plate.L)
    dt_g = plate.dt * ch.GRAIN_SUBSTEPS
    n_steps = int(t / dt_g)
    for _ in range(n_steps):
        g.step(amp, gx, gy, plate.dx, dt_g)
    return g, amp, gx, gy


def count_antinodes(amp, rel=0.5, sep=6):
    """Zarf alanındaki antinod tepe sayısı (yerel maksimumlar).

    Sınır hücrelerini sarmalamak (wrap) yanlış tepe üretir; dışarı -inf
    ile doldurulur.
    """
    thr = rel * amp.max()
    pad = np.full((amp.shape[0] + 2, amp.shape[1] + 2), -np.inf)
    pad[1:-1, 1:-1] = amp
    is_max = np.ones_like(amp, dtype=bool)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            is_max = is_max & (amp >= pad[di:di + amp.shape[0], dj:dj + amp.shape[1]])
    pts = np.argwhere(is_max & (amp > thr))
    picked = []
    for i, j in pts:
        if all((i - a) ** 2 + (j - b) ** 2 > sep * sep for a, b in picked):
            picked.append((i, j))
    return len(picked)


# ------------------------------------------------------------------- KAPILAR
def gate_g1_stability():
    """G1 kararlılık: |u| sınırlı, NaN yok, soğurucu enerji monoton azalıyor."""
    g = Gate("G1", "Soğurucu enerji monoton azalıyor, alan sınırlı ve NaN'sız")
    rng = np.random.default_rng(5)
    p = ch.Plate(n=GRID, damping=ch.DAMPING, drive_amp=0.0, freq=1.0,
                 boundary="neumann")
    p.u[1:-1, 1:-1] = 0.01 * rng.standard_normal((GRID - 2, GRID - 2))
    p.u_prev[1:-1, 1:-1] = p.u[1:-1, 1:-1]
    e0 = p.energy()
    energies = []
    for _ in range(16):
        p.run(1.0)
        energies.append(p.energy())
    energies = np.array(energies)
    inc = np.diff(energies)
    mono = bool(np.all(inc < 1e-9 * max(abs(e0), 1.0)))
    decay = energies[-1] / e0
    finite = bool(np.all(np.isfinite(p.u)))
    bounded = float(np.max(np.abs(p.u))) < 1.0
    g.check(mono and finite and bounded and decay < 0.10,
            f"E: {e0:.3e} -> {energies[-1]:.3e} (oran {decay:.2e}), "
            f"monoton={mono}, NaN yok={finite}, |u|max={np.max(np.abs(p.u)):.4f}")
    return g


def gate_g2_symmetry():
    """G2 simetri: kare plaka + merkez sürücü -> 4 katlı döndürme simetrisi."""
    g = Gate("G2", "Kare plaka + merkez sürücü 4 katlı döndürme simetrisi verir")
    p = settled_plate(FREQ_MODE0)
    amp = p.amplitude(normalize=True)
    err = ch.symmetry_error(amp)
    grains, _, _, _ = settled_grains(p)
    dens = ch.density_map(grains.p, 24, p.L)
    corr = ch.correlation(dens, np.rot90(dens))
    g.check(err["rot90"] < 0.02 and corr > 0.6,
            f"zarf rot90 hata={err['rot90']:.3e}, ayna hata={err['mirror']:.3e}, "
            f"kum yoğunluğu 4-katlı korelasyon={corr:.3f}")
    return g


def gate_g3_nodality():
    """G3 kum nodallik: taneler ortalama |u| düşük bölgelerde birikir."""
    g = Gate("G3", "Kum taneleri rastgele noktalara göre çok daha düşük |u| üzerinde")
    p = settled_plate(FREQ_MODE0)
    grains, amp, _, _ = settled_grains(p)
    a_grains = ch.bilinear(amp, grains.p, p.dx)
    rng = np.random.default_rng(99)
    rand = np.stack([rng.uniform(0, p.L, 4000), rng.uniform(0, p.L, 4000)], axis=1)
    a_rand = ch.bilinear(amp, rand, p.dx)
    ratio = float(a_grains.mean() / a_rand.mean())
    frac_dark = float((a_grains < 0.25 * a_rand.mean()).mean())
    g.check(ratio < 0.45 and frac_dark > 0.55,
            f"<|u|>_kum={a_grains.mean():.4f} / <|u|>_rastgele={a_rand.mean():.4f} "
            f"= {ratio:.3f} (oran), karanlık bölgedeki tane oranı={frac_dark:.2f}")
    return g


def gate_g4_pattern_change():
    """G4 desen değişimi: farklı sürücü frekansı -> farklı desen."""
    g = Gate("G4", "Farklı sürücü frekansı farklı nodal deseni üretir")
    p0 = settled_plate(FREQ_MODE0)
    p1 = settled_plate(FREQ_MODE1)
    a0 = p0.amplitude(normalize=True)
    a1 = p1.amplitude(normalize=True)
    corr = ch.correlation(a0, a1)
    n0 = count_antinodes(a0)
    n1 = count_antinodes(a1)
    cells0 = ch.nodal_cell_count(a0)
    cells1 = ch.nodal_cell_count(a1)
    g.check(corr < 0.75 and n0 != n1 and cells0 != cells1,
            f"zarf korelasyonu={corr:.3f}, antinod sayısı {n0} -> {n1}, "
            f"nodal hücre {cells0} -> {cells1}")
    return g


def gate_g5_determinism():
    """G5 determinizm: aynı tohum -> bit bit aynı tane konumları."""
    g = Gate("G5", "Aynı tohum aynı tane konumlarını bit bit üretir")
    p = settled_plate(FREQ_MODE1)
    g1, _, _, _ = settled_grains(p, seed=SEED)
    g2, _, _, _ = settled_grains(p, seed=SEED)
    g3, _, _, _ = settled_grains(p, seed=SEED + 1)
    same = np.array_equal(g1.p, g2.p)
    diff = not np.array_equal(g1.p, g3.p)
    maxdiff = float(np.max(np.abs(g1.p - g2.p)))
    g.check(same and diff and maxdiff == 0.0,
            f"aynı tohum: bit-bit aynı={same} (max fark {maxdiff:g}), "
            f"farklı tohum: farklı={diff}")
    return g


def gate_g6_bounds():
    """G6 sınırlar: plaka dışına kaçan tane yok, sayı sabit, konumlar sonlu."""
    g = Gate("G6", "Hiçbir tane plaka dışına çıkmaz; sayı sabit, konumlar sonlu")
    p = settled_plate(FREQ_MODE0)
    grains, _, _, _ = settled_grains(p)
    pos = grains.p
    inside = bool(np.all(pos >= 0.0) and np.all(pos <= p.L))
    margin = float(min(pos[:, 0].min(), pos[:, 1].min(),
                       p.L - pos[:, 0].max(), p.L - pos[:, 1].max()))
    n = pos.shape[0]
    finite = bool(np.all(np.isfinite(pos)))
    g.check(inside and finite and n == N_GRAINS and margin >= 0.0,
            f"tane sayısı={n}, plaka içinde={inside}, sonlu={finite}, "
            f"kenar payı={margin:.4f} L")
    return g


def gate_g7_resonance():
    """G7 rezonans: ölçülen tepe frekansı gerçek mod frekansına yakın."""
    g = Gate("G7", "Ölçülen rezonans frekansı analitik mod frekansıyla örtüşür")
    out = []
    ok = True
    for (m, n_m), f_ana in (((2, 0), FREQ_MODE0), ((2, 2), FREQ_MODE1)):
        f_meas, f_ana2, (fr, pw) = ch.find_mode_resonance(
            m, n_m, span=0.15, n_points=15, settle=20.0, grid_n=101,
            damping=0.4, boundary="neumann")
        err = abs(f_meas - f_ana) / f_ana
        out.append(f"({m},{n_m}): ölçülen {f_meas:.4f} Hz / analitik {f_ana2:.4f} Hz "
                   f"(%{100 * err:.3f} sapma)")
        ok = ok and err < 0.02
    g.check(ok, " | ".join(out))
    return g


# ------------------------------------------------------------------- ana
def main():
    t0 = time.time()
    print("=" * 78)
    print("CHLADNI KAPILARI — hepsi sayısal, hepsi yeşil olmalı")
    print("=" * 78)
    for fn in (gate_g1_stability, gate_g2_symmetry, gate_g3_nodality,
               gate_g4_pattern_change, gate_g5_determinism, gate_g6_bounds,
               gate_g7_resonance):
        fn()
    n_pass = sum(1 for _, _, ok, _ in _results if ok)
    print("-" * 78)
    print(f"SONUÇ: {n_pass}/{len(_results)} kapı geçti  "
          f"({time.time() - t0:.1f} s)")
    if n_pass != len(_results):
        for name, claim, ok, detail in _results:
            if not ok:
                print(f"  BAŞARISIZ: {name} — {claim} — {detail}")
        sys.exit(1)
    print("TÜM KAPILAR GEÇTİ ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
