"""
Chladni plaka fiziği — saf Python + numpy (Blender YOK, scipy YOK).

FİZİK ÇEKİRDEĞİ
================
1) PLAKA: 2B dalga denklemi, kare plaka üzerinde FDTD (leapfrog)

        u_tt = c² ∇²u  -  γ·u_t  +  f(x, y, t)

   - ∇²u: 5-noktalı stencil. Sınır koşulu: SERBEST KENAR (Neumann,
     ∂u/∂n = 0) — gerçek Chladni plakası gibi. Böylece plaka kenarları
     antinoddur ve kum yalnız İÇ nodal çizgilerde toplanır (kelepçeli kenar
     kullanılsaydı kenarlar da nodal olurdu ve tüm kum köşelere akardı).
   - γ: enerjiyi yutan (hava/sürtünme) soğurucu.
   - f: merkezde sürekli harmonik sürücü:  f = A·sin(ωt)·w(x,y), w = geniş
     Gauss "titrak" (σ = 0.13 L). Geniş sürücü düşük modları seçer, yüksek
     (ızgara ölçekli) modları sönümlendirir. Ayrıca rijit (0,0) modunun
     uyarılmaması için sürücünün uzaysal ortalaması sıfırlanır.
   - Zaman adımı: dt = 0.72 · dx/(c√2)   (CFL: c·dt/dx·√2 < 1)

   Leapfrog (sönümlü) güncellemesi:
        (1 + γ dt/2)·u⁺ = 2u - (1 - γ dt/2)·u⁻ + dt²( c²∇²u + f )
   `u⁻` = önceki zaman seviyesi, `u⁺` = yeni zaman seviyesi.
   (1 - γdt/2) çarpanı şarttır; atlanırsa etkin sönüm yarıya düşer.)

   Zaman-ortalı hız (u⁺-u⁻)/(2dt) enerji için kullanılır.

2) ANALİTİK MOD REFERANSI: serbest kenarlı kare plakanın modları
        φ_mn(x,y) = cos(mπx/L)·cos(nπy/L),  ω_mn = c·π·√(m²+n²)/L
        => f_mn = c·√(m²+n²)/(2L)   (Hz)
   Varsayılan c = √2, L = 1 için: f_11 = 1.000 Hz, f_20 = 1.4142 Hz,
   f_22 = 2.0000 Hz. FDTD ayrıklaması bunlara çok küçük (O(h²)) bir
   dispersiyon hatası ekler.

3) REZONANS TARAMASI: sürücü frekansı taranır. Noktasal genlik ölçümü
   modlar arası girişimle bozulduğu için MODAL PROJEKSİYON kullanılır
   (modlar ortogonaldir): ölçüt, soğurulan güç ∝ ω²⟨c_mn²⟩'dir ve tepe
   tam olarak ω₀'da yapar.

4) KUM (Chladni mekanizması): plaka titreşirken kum taneleri yüzeyden
   fırlatılır; her sıçrayış rastgele bir yönde olur ve adım uzunluğu
   yerel titreşim genliğiyle orantılıdır. Buna ek olarak tane, yerel
   titreşim genliğinin uzaysal gradyanı boyunca "yokuş aşağı" akar
   (enerji eğimi yönünde sürüklenme). Yerel genlik a_capture eşiğinin
   altına düşünce plaka taneyi artık fırlatmaz: tane düştüğü yerde
   YAPIŞIR (klasik Chladni yakalama ölçütü). Bu, kumun nodal çizgi
   boyunca çizgi kesişimlerine kayıp gitmesini engeller.

   Tane dinamiği (her alt-adımda):
        a(x,y)  = yerel titreşim genliği (zarf)
        ∇a      = genliğin gradyanı
        v += -σ·a·ξ        (ξ ~ N(0,1): plaka titreşiminin rastgele itkisi)
        v += -μ·∇a         (genlik eğimi boyunca yokuş aşağı akış)
        v *= exp(-ζ·dt)    (tane–plaka sürtünmesi)
        v += ε·ξ           (ortam gürültüsü)
        a < a_capture  ->  v = 0 (tane yapışır)
        p += v·dt
   Sınırda yansıma (plaka dışına kaçan tane yok); plaka kenarındaki
   sıkıştırma oluğu (margin) nedeniyle kum kenara birilemez.

   Zarf alanı: a(x,y) = √(2·⟨u²⟩) — sürücü altında u ≈ Re[U e^{iωt}].

Bu modül yalnızca numpy kullanır; Blender içinde de aynen çalışır.
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------- sabitler
GRID_N = 201          # kare plaka ızgarası (201x201 -> merkez tam hücre)
PLATE_L = 1.0         # plaka kenar uzunluğu (simülasyon birimi)
WAVE_C = math.sqrt(2.0)  # dalga hızı: (1,1) modu tam olarak 1.000 Hz olsun
DAMPING = 0.25        # γ: enerji soğurucu (1/s) -> Q = 2πf/γ
DRIVE_SIGMA = 0.13    # sürücü Gauss yarıçapı (L birimi) -- geniş "titrak"
CFL = 0.72            # dt = CFL · dx/(c√2)
DRIVE_AMP = 8.0       # sürücü kuvvet genliği

# tane dinamiği parametreleri (göreli birimler)
GRAIN_SIGMA = 3.0     # σ: rastgele itki şiddeti (titreşimle ölçeklenir)
GRAIN_THERMAL = 0.15  # σ0: ortam gürültüsü tabanı (nodal çizgide de kıpırdar)
GRAIN_MU = 0.08       # μ: genlik eğimi boyunca sürüklenme (yokuş aşağı)
GRAIN_ZETA = 8.0      # ζ: tane-plaka sürtünmesi (1/s)
GRAIN_SUBSTEPS = 5    # her kaç FDTD adımında bir tane güncellemesi
GRAIN_MARGIN = 0.035  # plaka kenarındaki sıkıştırma oluğu: kum içeride kalır
GRAIN_CAPTURE = 0.09  # yakalama eşiği: bundan düşük titreşimde tane yerinde kalır


# --------------------------------------------------------------- yardımcılar
def mode_frequency(m: int, n: int, L: float = PLATE_L, c: float = WAVE_C) -> float:
    """Sabit kenarlı kare plakanın (m,n) modunun frekansı (Hz).

        ω_mn = c·π·√(m²+n²)/L   (rad/s)  ->  f_mn = ω_mn/2π = c·√(m²+n²)/(2L)

    Varsayılan c = √2, L = 1 için:  f_11 = 1.000 Hz, f_13 = 2.236 Hz, f_33 = 3.000 Hz.
    """
    return c * math.sqrt(m * m + n * n) / (2.0 * L)


def rot90(a: np.ndarray) -> np.ndarray:
    """Kare diziyi merkez etrafında 90° döndür (karşı-el yönü).

    Izgarada N tek sayı olduğu için merkez hücre tam olarak yerinde kalır.
    """
    return np.rot90(a, 1).copy()


def mirror_x(a: np.ndarray) -> np.ndarray:
    """x eksenine göre aynala (merkez eksen)."""
    return a[:, ::-1].copy()


# ------------------------------------------------------------------- plaka
class Plate:
    """Kare plaka üzerinde FDTD (leapfrog) dalga çözücüsü.

    Dizin düzeni: u[i, j], i = x indeksi, j = y indeksi.
    """

    def __init__(self, n: int = GRID_N, L: float = PLATE_L, c: float = WAVE_C,
                 damping: float = DAMPING, drive_amp: float = 0.0,
                 freq: float = 0.0, drive_sigma: float = DRIVE_SIGMA,
                 env_tau: float = 0.55, boundary: str = "neumann"):
        self.n = int(n)
        self.L = float(L)
        self.c = float(c)
        self.damping = float(damping)
        self.dx = self.L / (self.n - 1)
        self.dt = CFL * self.dx / (self.c * math.sqrt(2.0))
        self.t = 0.0
        self.step_count = 0
        self.boundary = boundary

        self.u = np.zeros((self.n, self.n))
        self.u_prev = np.zeros((self.n, self.n))
        self.u_next = np.zeros((self.n, self.n))

        self.freq = float(freq)
        self.omega = 2.0 * math.pi * self.freq
        self.drive_amp = float(drive_amp)

        # sürücü ağırlık alanı (merkezde Gauss)
        ax = np.arange(self.n) * self.dx
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        cx = cy = 0.5 * self.L
        w = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * drive_sigma ** 2)))
        w /= w.max()
        if boundary == "neumann":
            # serbest kenar: rijit (0,0) modunun uyarılmaması için
            # sürücünün uzaysal ortalaması sıfırlanır (kaide tepki kuvveti alır)
            w = w - w.mean()
        self.w = w

        # zarf (running RMS) takibi: a = sqrt(2·⟨u²⟩)
        self.env_tau = float(env_tau)
        self._env_a = 1.0 - math.exp(-self.dt / self.env_tau)
        self.u2_ema = np.zeros((self.n, self.n))

    # ------------------------------------------------------------ tek adım
    def step(self) -> None:
        """Bir FDTD leapfrog adımı (konum-zaman ayrıklaması)."""
        n, dx2, dt2 = self.n, self.dx * self.dx, self.dt * self.dt
        u, up = self.u, self.u_prev

        if self.boundary == "neumann":
            # serbest kenar: ayna (reflect) hayalet hücreler -> du/dn = 0
            pad = np.pad(u, 1, mode="reflect")
            lap = (pad[0:n, 1:n + 1] + pad[2:n + 2, 1:n + 1]
                   + pad[1:n + 1, 0:n] + pad[1:n + 1, 2:n + 2] - 4.0 * u)
        else:
            # kelepçeli kenar: u = 0 (Dirichlet)
            lap = np.zeros_like(u)
            lap[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:]
                               + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1])

        f = self.drive_amp * math.sin(self.omega * self.t) * self.w

        # sönümlü leapfrog:
        #   (1+γdt/2)·u⁺ = 2u - (1-γdt/2)·u⁻ + dt²(c²∇²u + f)
        rhs = (2.0 * u - (1.0 - self.damping * self.dt / 2.0) * up
               + dt2 * (self.c * self.c * lap / dx2 + f))
        nxt = rhs / (1.0 + self.damping * self.dt / 2.0)

        un = self.u_next
        un[:, :] = nxt
        if self.boundary != "neumann":
            un[0, :] = 0.0
            un[-1, :] = 0.0
            un[:, 0] = 0.0
            un[:, -1] = 0.0

        # tampon rotasyonu
        self.u_prev, self.u, self.u_next = self.u, un, self.u_prev

        # zarf takibi
        a = self._env_a
        self.u2_ema *= (1.0 - a)
        self.u2_ema += a * self.u ** 2

        self.t += self.dt
        self.step_count += 1

    # ------------------------------------------------------------- yardımcılar
    def run(self, seconds: float) -> None:
        """Belirtilen simülasyon süresi kadar adım at."""
        n_steps = int(round(seconds / self.dt))
        for _ in range(n_steps):
            self.step()

    @property
    def envelope(self) -> np.ndarray:
        """Titreşim zarfı a = sqrt(2·⟨u²⟩) (normalize edilmemiş)."""
        return np.sqrt(2.0 * np.maximum(self.u2_ema, 0.0))

    def amplitude(self, normalize: bool = True) -> np.ndarray:
        """Zarf alanı; normalize=True ise maksimum 1 olacak şekilde ölçeklenir."""
        a = self.envelope
        if normalize:
            m = a.max()
            if m > 0:
                a = a / m
        return a

    def velocity(self) -> np.ndarray:
        """Zaman-ortalı hız (u⁺-u⁻)/(2dt)."""
        return (self.u - self.u_prev) / (2.0 * self.dt)

    def energy(self) -> float:
        """Toplam mekanik enerji: ½⟨u_t²⟩ + ½c²⟨|∇u|²⟩ (iç bölge)."""
        c = self.c
        dx = self.dx
        vt = self.velocity()
        kin = 0.5 * float(np.sum(vt[1:-1, 1:-1] ** 2))
        ux = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * dx)
        uy = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * dx)
        pot = 0.5 * c * c * float(np.sum(ux ** 2 + uy ** 2))
        return kin + pot

    def switch_to(self, freq: float, drive_amp: float) -> None:
        """Sürücüyü anında değiştir (mod değişimi: "müzik değişti")."""
        self.freq = float(freq)
        self.omega = 2.0 * math.pi * self.freq
        self.drive_amp = float(drive_amp)


# ------------------------------------------------------- rezonans taraması
def mode_basis_vector(n: int, m: int, boundary: str = "neumann") -> np.ndarray:
    """m. modun 1B taban vektörü (iç hücreler).

    Dirichlet (kelepçeli kenar):  φ_m(i) = sin(mπi/(N-1))
    Neumann  (serbest kenar):    φ_m(i) = cos(mπi/(N-1))   [m=0 -> rijit mod]
    """
    i = np.arange(1, n - 1)
    if boundary == "neumann":
        return np.cos(m * math.pi * i / (n - 1))
    return np.sin(m * math.pi * i / (n - 1))


def modal_coefficient(u: np.ndarray, sm: np.ndarray, sn: np.ndarray) -> float:
    """Alanın (m,n) modu üzerine projeksiyonu: c_mn = φ_mᵀ U φ_n."""
    return float(sm @ u[1:-1, 1:-1] @ sn)


def scan_mode(m: int, n_m: int, freqs, settle: float = 8.0,
              grid_n: int = GRID_N, damping: float = 1.0,
              drive_amp: float = DRIVE_AMP, c: float = WAVE_C,
              boundary: str = "neumann") -> np.ndarray:
    """Tek bir modun frekans yanıtını ölç (soğurulan güç ∝ ω²⟨c²⟩).

    Noktasal ölçüm yerine MODAL projeksiyon kullanılır: modlar birbirine
    dik (ortogonal) olduğu için farklı modların girişimleri ölçümü bozmaz.
    Soğurulan güç ω²⟨c²⟩, sönümlü tek serbestlikli sistemde tam olarak ω₀'da
    tepe yapar (rezonans frekansı kaymaz).
    """
    sm = mode_basis_vector(grid_n, m, boundary)
    sn = mode_basis_vector(grid_n, n_m, boundary)
    out = np.zeros(len(freqs))
    for k, fr in enumerate(freqs):
        fr = float(fr)
        p = Plate(n=grid_n, c=c, damping=damping, drive_amp=drive_amp, freq=fr,
                  boundary=boundary)
        p.run(settle)
        n_acc = max(int((1.0 / fr) / p.dt), 8)
        acc = 0.0
        for _ in range(n_acc):
            p.step()
            cc = modal_coefficient(p.u, sm, sn)
            acc += cc * cc
        out[k] = fr * fr * acc / n_acc
    return out


def peak_frequency(freqs, values) -> float:
    """Tepe frekansı: en yüksek nokta + parabolik (üç noktalı) arıma."""
    f = np.asarray(freqs, dtype=float)
    v = np.asarray(values, dtype=float)
    k = int(np.argmax(v))
    if 0 < k < len(f) - 1:
        y = np.log(np.maximum(v, 1e-300))
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den < -1e-12:
            delta = 0.5 * (y0 - y2) / den
            delta = float(np.clip(delta, -1.0, 1.0))
            return float(f[k] + delta * (f[k + 1] - f[k - 1]) / 2.0)
    return float(f[k])


def find_mode_resonance(m: int, n_m: int, span: float = 0.15, n_points: int = 15,
                         settle: float = 20.0, grid_n: int = 101,
                         damping: float = 0.4, drive_amp: float = DRIVE_AMP,
                         c: float = WAVE_C, L: float = PLATE_L,
                         boundary: str = "neumann"):
    """Bir modun rezonans frekansını simülasyonla ölç.

    `span` analitik frekansın çevresindeki göreli tarama yarıçapıdır.
    Dönen: (ölçülen tepe frekansı, analitik frekans, (f, güç) dizisi).
    """
    f_ana = mode_frequency(m, n_m, L=L, c=c)
    freqs = np.linspace(f_ana * (1.0 - span), f_ana * (1.0 + span), n_points)
    power = scan_mode(m, n_m, freqs, settle=settle, grid_n=grid_n,
                      damping=damping, drive_amp=drive_amp, c=c, boundary=boundary)
    return peak_frequency(freqs, power), f_ana, (freqs, power)


def strongest_mode(freqs, power_matrix, modes) -> tuple:
    """Mod güç matrisinden en güçlü tepkiyi (mod, frekans) olarak ver."""
    best = (-1.0, None, None)
    for idx, (m, n_m) in enumerate(modes):
        pk = peak_frequency(freqs, power_matrix[idx])
        val = float(np.max(power_matrix[idx]))
        if val > best[0]:
            best = (val, (m, n_m), pk)
    return best[1], best[2]



# ------------------------------------------------------------- kum taneleri
def bilinear(field: np.ndarray, p: np.ndarray, dx: float) -> np.ndarray:
    """(n,2) konum listesinde bilineer örnekleme; (n,) sonuç."""
    n = field.shape[0]
    fx = np.clip(p[:, 0] / dx, 0.0, n - 1.001)
    fy = np.clip(p[:, 1] / dx, 0.0, n - 1.001)
    i0 = fx.astype(np.int64)
    j0 = fy.astype(np.int64)
    tx = fx - i0
    ty = fy - j0
    v00 = field[i0, j0]
    v10 = field[i0 + 1, j0]
    v01 = field[i0, j0 + 1]
    v11 = field[i0 + 1, j0 + 1]
    return ((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10
            + (1 - tx) * ty * v01 + tx * ty * v11)


def bilinear_vec(fx: np.ndarray, fy: np.ndarray, p: np.ndarray,
                 dx: float) -> np.ndarray:
    """İki bileşenli alanın (n,2) konumlarında bilineer örneklemesi."""
    n = fx.shape[0]
    gx = np.clip(p[:, 0] / dx, 0.0, n - 1.001)
    gy = np.clip(p[:, 1] / dx, 0.0, n - 1.001)
    i0 = gx.astype(np.int64)
    j0 = gy.astype(np.int64)
    tx = gx - i0
    ty = gy - j0
    out = np.empty((p.shape[0], 2))
    for k, f in enumerate((fx, fy)):
        v00 = f[i0, j0]
        v10 = f[i0 + 1, j0]
        v01 = f[i0, j0 + 1]
        v11 = f[i0 + 1, j0 + 1]
        out[:, k] = ((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10
                     + (1 - tx) * ty * v01 + tx * ty * v11)
    return out


class GrainField:
    """Chladni kumu: tane konumları + titreşim-zarfına bağlı dinamik.

    Tane dinamik denklemi (her alt-adımda):
        v += -(σ·a + σ0)·ξ          titreşim itkisi + ortam gürültüsü (ξ ~ N(0,1))
        v += -μ·∇a                  genlik eğimi boyunca "yokuş aşağı" akış
        v *= exp(-ζ·dt)             tane-plaka sürtünmesi
        p += v·dt
    """

    def __init__(self, n_grains: int = 2400, seed: int = 20240517,
                 L: float = PLATE_L, margin: float = GRAIN_MARGIN):
        self.L = float(L)
        self.margin = float(margin)
        self.n_grains = int(n_grains)
        self.rng = np.random.default_rng(seed)
        self.p = np.stack([
            self.rng.uniform(margin, self.L - margin, self.n_grains),
            self.rng.uniform(margin, self.L - margin, self.n_grains)], axis=1)
        self.v = np.zeros((self.n_grains, 2))
        self.seed = int(seed)

    def copy(self) -> "GrainField":
        g = GrainField.__new__(GrainField)
        g.L = self.L
        g.margin = self.margin
        g.n_grains = self.n_grains
        g.rng = np.random.default_rng(self.seed)
        g.p = self.p.copy()
        g.v = self.v.copy()
        g.seed = self.seed
        return g

    def step(self, amp: np.ndarray, gx: np.ndarray, gy: np.ndarray, dx: float,
             dt: float, sigma: float = GRAIN_SIGMA, mu: float = GRAIN_MU,
             zeta: float = GRAIN_ZETA, thermal: float = GRAIN_THERMAL,
             capture: float = GRAIN_CAPTURE) -> None:
        """Bir tane alt-adımı.

        amp: normalize zarf alanı (0..1), gx/gy: onun gradyanı.

        Chladni YAKALAMA ölçütü: yerel titreşim a_capture eşiğinin altına
        düşünce plaka taneyi artık fırlatmaz; tane düştüğü yerde kalır
        (klasik Chladni koşulu). Bu, kumun nodal çizgi boyunca "kayıp"
        gitmesini engeller; çizgi, çevreden gelen tanelerle boyanır.
        """
        a = bilinear(amp, self.p, dx)
        ga = bilinear_vec(gx, gy, self.p, dx)

        # yakalama: yerel titreşim eşiğin altındaysa tane düştüğü yerde kalır
        caught = a < capture
        rel = np.clip((a - capture) / max(capture, 1e-9), 0.0, 1.0)

        # (1) plaka titreşiminin rastgele itkisi (tane yüzeyden fırlatılır)
        self.v += (-sigma * a * rel)[:, None] \
            * self.rng.standard_normal(self.p.shape) * dt
        # (2) genlik eğimi boyunca "yokuş aşağı" akış (nodal çizgiye sürükler)
        self.v += (-mu * rel)[:, None] * ga * dt
        # (3) tane-plaka sürtünmesi
        self.v *= math.exp(-zeta * dt)
        # (4) ortam gürültüsü: yakalanmış tane bile mikro-kıpırdanır
        self.v += thermal * self.rng.standard_normal(self.p.shape) * dt
        # (5) yakalanan tane tamamen durur (nodal çizgiye yapışır)
        self.v[caught] = 0.0
        # (6) hareket
        self.p += self.v * dt

        # (5) plaka sınırı: yansıma (dışarıda kalan tane yok)
        lo, hi = self.margin, self.L - self.margin
        for k in range(2):
            below = self.p[:, k] < lo
            above = self.p[:, k] > hi
            self.p[below, k] = lo + (lo - self.p[below, k])
            self.p[above, k] = hi - (self.p[above, k] - hi)
            self.v[below | above, k] *= -1.0
        np.clip(self.p, lo, hi, out=self.p)


# ------------------------------------------------------------------ sahne
def _grad2(a: np.ndarray, dx: float):
    """Merkezi fark gradyanı -> (d/dx, d/dy).

    Dizin düzeni: a[i, j] -> i = x, j = y; np.gradient önce eksen 0 (x) verir.
    """
    gx, gy = np.gradient(a, dx, dx)
    return gx, gy


def simulate_scene(freq_a: float, freq_b: float, n_frames: int = 49,
                   n_grains: int = 2400, seed: int = 20240517,
                   n: int = GRID_N, L: float = PLATE_L, c: float = WAVE_C,
                   damping: float = DAMPING, drive_amp: float = DRIVE_AMP,
                   t_a: float = 20.0, t_b: float = 34.0,
                   window_a: float = 7.0, window_b: float = 5.0,
                   plate_switch_frame: int = 24, boundary: str = "neumann",
                   env_tau: float = 1.5, progress=None):
    """Tek bir sürekli simülasyon: Mod 0 -> Mod 1 geçişi + kum.

    Zaman eşlemesi (sinematik zaman sıkıştırması -- görüntünün tamamı
    simülasyonun çözümüdür, yalnızca oynatma hızı değiştirilir):
      kare 1..plate_switch_frame  -> sim t ∈ [t_a-window_a, t_a]  (Mod 0)
      kare plate_switch_frame+1..30-> sim t ∈ [t_a, t_a+8]        (kum göçü)
      kare 31..40                  -> sim t ∈ [t_a+8, t_a+t_b]    (plaka yeni moda geçer)
      kare 41..n_frames            -> sim t ∈ [t_a+t_b, t_a+t_b+window_b]

    Dönen sözlük:
        'times'  : her kare için simülasyon zamanı
        'grains' : (n_frames, n_grains, 2) tane konumları
        'amp'    : (n_frames, n, n) normalize zarf alanları
        'plate'  : FDTD plaka nesnesi (son hali)
    """
    f_switch = int(plate_switch_frame)
    knots = [
        (1, t_a - window_a),
        (f_switch, t_a),
        (f_switch + 6, t_a + 8.0),
        (n_frames - 9, t_a + t_b),
        (n_frames, t_a + t_b + window_b),
    ]
    times = np.zeros(n_frames)
    for i in range(n_frames):
        f = i + 1
        if f <= knots[0][0]:
            times[i] = knots[0][1]
            continue
        for (f0, t0), (f1, t1) in zip(knots[:-1], knots[1:]):
            if f0 <= f <= f1:
                w = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
                times[i] = t0 + w * (t1 - t0)
                break
        else:
            times[i] = knots[-1][1]

    plate = Plate(n=n, L=L, c=c, damping=damping, drive_amp=drive_amp,
                  freq=freq_a, env_tau=env_tau, boundary=boundary)
    grains = GrainField(n_grains=n_grains, seed=seed, L=L)

    grains_out = np.zeros((n_frames, n_grains, 2))
    amp_out = np.zeros((n_frames, n, n), dtype=np.float32)

    gx = np.zeros((n, n))
    gy = np.zeros((n, n))
    next_frame = 0
    t_switch = t_a

    while next_frame < n_frames:
        target = float(times[next_frame])
        if plate.t >= t_switch and plate.freq != freq_b:
            plate.switch_to(freq_b, drive_amp)
        while plate.t < target:
            plate.step()
            if plate.step_count % GRAIN_SUBSTEPS == 0:
                amp = plate.amplitude(normalize=True)
                gx, gy = _grad2(amp, plate.dx)
                grains.step(amp, gx, gy, plate.dx, plate.dt * GRAIN_SUBSTEPS)
        amp = plate.amplitude(normalize=True)
        gx, gy = _grad2(amp, plate.dx)
        grains.step(amp, gx, gy, plate.dx, plate.dt * GRAIN_SUBSTEPS)
        grains_out[next_frame] = grains.p
        amp_out[next_frame] = amp.astype(np.float32)
        if progress:
            progress(next_frame + 1, n_frames)
        next_frame += 1

    return {"times": times, "grains": grains_out, "amp": amp_out,
            "plate": plate, "freq_a": freq_a, "freq_b": freq_b}


# ------------------------------------------------------------- ölçümler
def nodal_cell_count(amp: np.ndarray, rel: float = 0.12) -> int:
    """Zarf alanının nodal bölgesine düştüğü hücre sayısı (ken şeridi hariç)."""
    n = amp.shape[0]
    k = int(0.06 * n)
    core = amp[k:n - k, k:n - k]
    thr = rel * amp.max()
    return int(np.sum(core < thr))


def crossing_count(u_line: np.ndarray) -> int:
    """1B örnek dizisindeki işaret değişimlerinin sayısı (gürültü toleranslı)."""
    v = np.asarray(u_line, dtype=float)
    m = np.max(np.abs(v))
    if m <= 0:
        return 0
    s = np.sign(v)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    # ölü bölge: iki taraf da neredeyse sıfırsa sayma
    out = 0
    for i in idx:
        if max(abs(v[i]), abs(v[i + 1])) > 1e-4 * m:
            out += 1
    return out


def line_samples(plate: Plate, frac: float = 0.5, n_samples: int = 401):
    """Plaka ortasından geçen yatay/dikey çizgiden alınan u örnekleri."""
    m = plate.n // 2
    row = plate.u[m, :]
    col = plate.u[:, m]
    return row, col


def symmetry_error(a: np.ndarray) -> dict:
    """4 katlı döndürme + ayna simetrisi hata ölçüleri (normalize RMS)."""
    a = np.asarray(a, dtype=float)
    den = np.sqrt(np.sum(a * a))
    if den <= 0:
        return {"rot90": 1.0, "mirror": 1.0}
    e90 = np.sqrt(np.sum((a - rot90(a)) ** 2)) / den
    em = np.sqrt(np.sum((a - mirror_x(a)) ** 2)) / den
    return {"rot90": float(e90), "mirror": float(em)}


def density_map(p: np.ndarray, n_bins: int = 24, L: float = PLATE_L) -> np.ndarray:
    """Tane konumlarından n×n yoğunluk haritası (ken şeridi atlanır)."""
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=n_bins,
                             range=[[0, L], [0, L]])
    k = 1
    return h[k:n_bins - k, k:n_bins - k]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson korelasyonu."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= 0:
        return 0.0
    return float(np.sum(a * b) / den)
