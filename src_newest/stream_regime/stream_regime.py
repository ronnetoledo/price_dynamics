"""Detector espectral de regime online/causal (tick-a-tick) para HFT.

Implementa o design de docs/stream_regime.md: a contraparte streaming do
build_causal_regime_matrix. Estado incremental atualizado a cada barra em
O(embed**2 * p), sem refazer eigh(C) do zero.

Três camadas:
  (1) resíduo de inovação ê_t  — latência 1 tick, arma WATCH
  (2) subspace iteration warm-started — rastreia top-p modos, sinais espectrais
  (3) CUSUM/Page-Hinkley nos sinais  — confirma a mudança (CONFIRMED)

Covariância via EWMA (rank-1 puro, sempre PSD) em vez de sliding window duro.

Uso como harness de replay:
    python stream_regime.py --symbol PETR4 --tf M1 --embed 70 --alpha 0.02
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # núcleo na raiz de src_newest

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, subspace_angles


def mp_lambda_plus(var: float, q: float) -> float:
    """Limite superior de Marchenko-Pastur."""
    return var * (1.0 + np.sqrt(q)) ** 2


class PageHinkley:
    """CUSUM bilateral sobre um sinal padronizado por baseline EWMA.

    O baseline (média/variância correntes) usa um alpha lento para não absorver
    a própria mudança que se quer detectar. z = (y - média)/desvio; o acumulador
    dispara quando |g| ultrapassa h. Reinicia no disparo.
    """

    def __init__(self, h: float = 8.0, nu: float = 0.5, alpha_base: float = 0.01):
        self.h = h
        self.nu = nu
        self.alpha_base = alpha_base
        self.mean = None
        self.var = 1.0
        self.g_pos = 0.0
        self.g_neg = 0.0

    def update(self, y: float) -> tuple[bool, float]:
        if self.mean is None:        # primeira amostra: só inicializa o baseline
            self.mean = y
            return False, 0.0
        z = (y - self.mean) / np.sqrt(max(self.var, 1e-12))
        self.g_pos = max(0.0, self.g_pos + z - self.nu)
        self.g_neg = max(0.0, self.g_neg - z - self.nu)
        alarm = self.g_pos > self.h or self.g_neg > self.h
        if alarm:
            self.g_pos = self.g_neg = 0.0
        # atualiza baseline DEPOIS de avaliar z (z usa só o passado)
        d = y - self.mean
        self.mean += self.alpha_base * d
        self.var = (1.0 - self.alpha_base) * (self.var + self.alpha_base * d * d)
        return alarm, max(self.g_pos, self.g_neg)


@dataclass
class RegimeState:
    t: int
    state: str                       # NORMAL | WATCH | CONFIRMED
    e_resid: float                   # resíduo de inovação ê_t (fração de energia)
    novelty: float                   # energia residual / nível do bulk (~1 sob ruído)
    trace: float                     # tr C (= vol**2 * d)
    lam1: float                      # maior autovalor
    m: int                           # nº de modos acima de λ₊
    entropy: float                   # entropia espectral normalizada
    lam_plus: float                  # limiar Marchenko-Pastur
    theta: float                     # maior ângulo de subespaço vs t-τ
    drift: float                     # ⟨r⟩ líquido recente (momentum bruto, ímpar)
    persist: float                   # autocorr lag-1 (>0 persistente, <0 reverte)
    direction: float                 # sinal direcional condicionado ∈ [-1,1]
    trendiness: float                # λ₁/trace do nível (SSA): força da tendência
    alarms: list = field(default_factory=list)
    eigenvalues: np.ndarray = None


class StreamRegimeDetector:
    """Detector espectral incremental. Um por ativo. Chame update(r) por tick."""

    def __init__(self, embed: int, p: int = 8, alpha: float = 0.02,
                 h: float = 15.0, nu: float = 1.0,
                 watch_decay: int = 30, confirm_signals=("m", "entropy"),
                 direction_mode: str = "auto", direction_source: str = "drift",
                 conf_gate: str = "entropy", ssa_window: int | None = None,
                 track_rotation: bool = True,
                 rotation_lag: int = 20, reanchor_every: int = 1000,
                 n_warmup: int | None = None):
        if p >= embed:
            raise ValueError("p deve ser < embed (precisa de bulk para σ²)")
        self.track_rotation = track_rotation   # subspace_angles é caro; desligar em lote
        self.m_dim = embed
        self.p = p
        self.alpha = alpha
        self.watch_decay = watch_decay
        # 'auto' = persistência (autocorr) decide momentum/reversão; ou força um
        self.direction_mode = direction_mode
        # fonte do sinal linear de direção: 'drift' (retorno líquido) ou 'ssa'
        # (inclinação da reconstrução SSA do nível log-preço — denoised, gauge-free)
        self.direction_source = direction_source
        # gate de confiança: 'entropy' = (1−S); 'trend' = (1−S)·trendiness do nível
        self.conf_gate = conf_gate
        self.ssa_window = ssa_window if ssa_window is not None else max(3, embed // 3)
        K = embed - self.ssa_window + 1
        # pesos da média diagonal (hankelização) da reconstrução de posto 1
        self._ssa_counts = np.array(
            [min(t + 1, self.ssa_window, K, embed - t) for t in range(embed)],
            dtype=float)
        # sinais que confirmam (camada 3). trace/lam1 são vol — atrasados, ficam de
        # fora do gatilho; m/entropia são estruturais e antecipam o breakout.
        self.confirm_signals = set(confirm_signals)
        self.rotation_lag = rotation_lag
        self.reanchor_every = reanchor_every
        self.n_warmup = n_warmup if n_warmup is not None else 5 * embed

        # n_eff de uma EWMA com peso alpha; q efetivo entra no limiar MP
        self.n_eff = (2.0 - alpha) / alpha
        self.q = embed / self.n_eff

        self.buf = np.zeros(embed)        # últimos `embed` retornos (snapshot sₙ)
        self.n_buf = 0
        self.mu = np.zeros(embed)         # média EWMA do snapshot
        self.C = np.zeros((embed, embed)) # covariância EWMA
        self.Q = None                     # (embed, p) top-p autovetores
        self.Lambda = None                # (p,) autovalores rastreados
        self.sigma2 = None                # σ² do bulk (do tick anterior)
        self.t = 0

        self._Q_hist: list[np.ndarray] = []   # ring de Q para o ângulo de subespaço
        self._cusum = {k: PageHinkley(h, nu) for k in
                       ("resid", "trace", "lam1", "m", "entropy")}
        self._watch_age = None
        self.state = "NORMAL"

    # ── núcleo incremental ────────────────────────────────────────────────────
    def _push_return(self, r: float):
        self.buf = np.roll(self.buf, -1)
        self.buf[-1] = r
        self.n_buf = min(self.n_buf + 1, self.m_dim)

    def _ewma_update(self, s: np.ndarray):
        """EWMA de média e covariância (West). Rank-1 puro → sempre PSD."""
        a = self.alpha
        delta = s - self.mu
        self.mu += a * delta
        # C_t = (1-a)(C_{t-1} + a·δδᵀ)
        self.C *= (1.0 - a)
        self.C += (1.0 - a) * a * np.outer(delta, delta)

    def _subspace_step(self, iters: int = 1):
        """Iteração de subespaço warm-started: 1 passo basta (C_t ≈ C_{t-1})."""
        Q = self.Q
        for _ in range(iters):
            Z = self.C @ Q
            Q, _ = np.linalg.qr(Z)
        H = Q.T @ (self.C @ Q)
        theta, U = eigh(H)
        order = np.argsort(theta)[::-1]
        self.Lambda = theta[order]
        self.Q = Q @ U[:, order]

    def _full_anchor(self):
        """Re-ancoragem: eigh completo de C, fica com os top-p (zera deriva)."""
        vals, vecs = eigh(self.C)
        order = np.argsort(vals)[::-1][:self.p]
        self.Lambda = vals[order]
        self.Q = vecs[:, order]

    # ── sinais espectrais ──────────────────────────────────────────────────────
    def _sigma2_bulk(self) -> float:
        """Média dos autovalores do bulk = (tr C − Σλ_top) / (embed − p).

        Estima σ² da MP sem precisar dos autovalores do bulk individualmente
        (que não rastreamos). Válido se p ≥ nº real de modos estruturais.
        """
        bulk_mass = np.trace(self.C) - self.Lambda.sum()
        return max(bulk_mass / (self.m_dim - self.p), 1e-15)

    def _entropy(self, sigma2: float) -> float:
        """Entropia espectral normalizada, com o bulk representado por (embed−p)
        autovalores iguais a σ². Aproxima o espectro completo de forma consistente
        com o traço."""
        bulk = np.full(self.m_dim - self.p, sigma2)
        lam = np.concatenate([np.maximum(self.Lambda, 0.0), bulk])
        lam = lam[lam > 0]
        pk = lam / lam.sum()
        return float(-np.sum(pk * np.log(pk)) / np.log(self.m_dim))

    def _subspace_angle(self) -> float:
        """Maior ângulo principal entre Q_t e Q de `rotation_lag` ticks atrás."""
        if len(self._Q_hist) <= self.rotation_lag:
            return 0.0
        Q_old = self._Q_hist[-self.rotation_lag - 1]
        return float(subspace_angles(self.Q, Q_old)[0])

    def _ssa_level(self, s: np.ndarray):
        """SSA local do nível log-preço (= cumsum dos retornos da janela).

        Decompõe a Hankel do nível, reconstrói o modo líder (tendência) por média
        diagonal e devolve (inclinação, trendiness). A reconstrução de posto 1
        `a₁·v₁ᵀ` é **gauge-free** (o sinal de v₁ cancela), então a inclinação tem
        sinal bem-definido — sem o problema de gauge do sign(v₁).
        """
        level = np.cumsum(s)
        level = level - level.mean()
        w, N = self.ssa_window, self.m_dim
        K = N - w + 1
        st = level.strides[0]
        T = np.lib.stride_tricks.as_strided(level, (K, w), (st, st))
        vals, vecs = eigh(T.T @ T)               # w×w, pequeno
        v1 = vecs[:, -1]
        trendiness = float(vals[-1] / vals.sum()) if vals.sum() > 0 else 0.0
        recon = np.convolve(T @ v1, v1) / self._ssa_counts   # hankelização posto 1
        x = np.arange(N) - (N - 1) / 2.0
        slope = float(x @ recon / (x @ x))
        return slope, trendiness

    # ── direção (objetos ímpares sob r→−r, condicionados ao regime) ─────────────
    def _direction(self, s: np.ndarray, m: int, entropy: float):
        """Sinal direcional gauge-safe condicionado ao regime.

        Linear/ímpar: 'drift' = retorno líquido recente, ou 'ssa' = inclinação da
        reconstrução SSA do nível (denoised). A persistência (autocorr lag-1) decide
        momentum (+) vs reversão (−); entropia/m dão a confiança. NUNCA usa sign(vₖ)
        cru — só objetos ímpares ou projetores (invariantes ao gauge).
        """
        drift = float(s.sum())                       # net log-return das últimas `embed` barras
        d = s - s.mean()
        denom = float(d[:-1] @ d[:-1])
        rho1 = float(d[1:] @ d[:-1] / denom) if denom > 1e-24 else 0.0

        # SSA do nível necessário se a fonte é SSA ou se o gate usa trendiness
        trendiness = 0.0
        ssa_slope = None
        if self.direction_source == "ssa" or self.conf_gate == "trend":
            ssa_slope, trendiness = self._ssa_level(s)

        if self.direction_source == "ssa":
            lin = ssa_slope
        else:
            # 'drift', com fallback p/ inclinação do retorno filtrado estrutural
            slope = 0.0
            if m >= 1:
                Vs = self.Q[:, :m]
                r_hat = Vs @ (Vs.T @ (s - self.mu))
                path = np.cumsum(r_hat)
                x = np.arange(self.m_dim) - (self.m_dim - 1) / 2.0
                slope = float(x @ path / (x @ x))
            lin = drift if abs(drift) >= abs(slope) else slope

        if self.direction_mode == "momentum":
            sgn = 1.0
        elif self.direction_mode == "reversion":
            sgn = -1.0
        else:                                        # 'auto': persistência decide
            sgn = np.sign(rho1) if rho1 != 0 else 1.0

        raw = sgn * np.sign(lin)
        conf = (1.0 - entropy) if m >= 1 else 0.0    # confiança ∈ [0,1]
        if self.conf_gate == "trend":
            conf *= trendiness                       # aposta direção só se o nível tende
        return drift, rho1, float(raw * conf), trendiness

    # ── ciclo por tick ──────────────────────────────────────────────────────────
    def update(self, r: float) -> RegimeState | None:
        self.t += 1
        self._push_return(r)
        if self.n_buf < self.m_dim:               # buffer ainda enchendo
            return None
        s = self.buf.copy()

        # CAMADA 1 — resíduo de inovação (ANTES de atualizar C, usa Q/σ² atuais).
        # e_resid é a fração de energia fora do subespaço (interpretável, mas
        # satura perto de 1 com bulk grande); novelty normaliza a energia residual
        # pelo nível esperado do bulk → ~1 sob ruído puro, salta na novidade.
        e_resid = 0.0
        novelty = 1.0
        if self.Q is not None:
            sc = s - self.mu
            norm2 = float(sc @ sc)
            if norm2 > 1e-24:
                proj = self.Q @ (self.Q.T @ sc)
                resid2 = float((sc - proj) @ (sc - proj))
                e_resid = np.sqrt(resid2 / norm2)
                expected = (self.m_dim - self.p) * self.sigma2
                novelty = resid2 / max(expected, 1e-24)

        # atualiza covariância EWMA
        self._ewma_update(s)

        # warmup: acumula C; inicializa Q só quando há amostra suficiente
        if self.Q is None:
            if self.t >= self.n_warmup:
                self._full_anchor()
                self.sigma2 = self._sigma2_bulk()
            return None

        # CAMADA 2 — tracker espectral
        if self.t % self.reanchor_every == 0:
            self._full_anchor()
        else:
            self._subspace_step()

        if self.track_rotation:
            self._Q_hist.append(self.Q.copy())
            if len(self._Q_hist) > self.rotation_lag + 2:
                self._Q_hist.pop(0)

        # sinais
        self.sigma2 = self._sigma2_bulk()
        lam_plus = mp_lambda_plus(self.sigma2, self.q)
        m = int(np.sum(self.Lambda > lam_plus))
        trace = float(np.trace(self.C))
        lam1 = float(self.Lambda[0])
        entropy = self._entropy(self.sigma2)
        theta = self._subspace_angle() if self.track_rotation else 0.0
        drift, persist, direction, trendiness = self._direction(s, m, entropy)

        # CAMADA 3 — change-point por sinal (resíduo = novelty normalizada)
        alarms = []
        for key, val in (("resid", novelty), ("trace", trace), ("lam1", lam1),
                         ("m", float(m)), ("entropy", entropy)):
            fired, _ = self._cusum[key].update(val)
            if fired:
                alarms.append(key)

        # fusão / máquina de estados, dirigida pelos próprios alarmes CUSUM
        resid_alarm = "resid" in alarms
        spectral_alarm = any(a in self.confirm_signals for a in alarms)

        # O sinal estrutural (m/entropia) lidera o breakout (~22 ticks no NVDA M1),
        # então confirma DIRETO; a novidade é coincidente e serve só de pré-arme.
        if self.state == "NORMAL":
            if spectral_alarm:
                self.state = "CONFIRMED"
            elif resid_alarm:               # pré-arme: novidade sem confirmação ainda
                self.state, self._watch_age = "WATCH", 0
        elif self.state == "WATCH":
            self._watch_age += 1
            if spectral_alarm:
                self.state = "CONFIRMED"
            elif self._watch_age > self.watch_decay:
                self.state = "NORMAL"
        elif self.state == "CONFIRMED":
            if not (resid_alarm or spectral_alarm):
                self.state = "NORMAL"

        return RegimeState(
            t=self.t, state=self.state, e_resid=e_resid, novelty=novelty,
            trace=trace, lam1=lam1, m=m, entropy=entropy, lam_plus=lam_plus,
            theta=theta, drift=drift, persist=persist, direction=direction,
            trendiness=trendiness, alarms=alarms, eigenvalues=self.Lambda.copy(),
        )


# ── harness de replay sobre Parquet ───────────────────────────────────────────
def replay(ret_C: np.ndarray, **kwargs) -> list[RegimeState]:
    """Roda o detector sobre uma série de log-retornos, devolve estados não-None."""
    det = StreamRegimeDetector(**kwargs)
    return [st for r in ret_C if (st := det.update(float(r))) is not None]


def confirmed_events(states: list[RegimeState]) -> list[RegimeState]:
    """Estados em que se entra em CONFIRMED (deduplica disparos consecutivos)."""
    out, prev = [], -999
    for st in states:
        if st.state == "CONFIRMED" and st.t - prev > 1:
            out.append(st)
        if st.state == "CONFIRMED":
            prev = st.t
    return out


def vol_breakouts(ret: np.ndarray, w: int = 60, ratio: float = 2.0,
                  refractory: int | None = None) -> np.ndarray:
    """Ground-truth: onsets de breakout de volatilidade realizada.

    Marca t onde a vol forward (próximas w barras) supera `ratio`× a vol trailing
    (w barras anteriores). Mantém só o início de cada episódio (refratário) para
    não contar a mesma rampa várias vezes.
    """
    if refractory is None:
        refractory = w
    r2 = ret ** 2
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    n = len(ret)

    def rv(a, b):                       # raiz da soma de quadrados em [a,b)
        return np.sqrt(csum[b] - csum[a])

    onsets, last = [], -10**9
    for t in range(w, n - w):
        trail = rv(t - w, t)
        fwd   = rv(t, t + w)
        if trail > 0 and fwd / trail > ratio and t - last > refractory:
            onsets.append(t)
            last = t
    return np.array(onsets, dtype=int)


def score(events: list[RegimeState], gt: np.ndarray,
          lead_max: int = 60, tol: int = 15) -> dict:
    """Pontua eventos CONFIRMED contra onsets ground-truth.

    Um onset gt é detectado se algum evento cai em [gt − lead_max, gt + tol].
    lead = gt − t_evento (positivo = antecipou). Precisão = eventos casados / total.
    """
    ev = np.array([e.t for e in events], dtype=int)
    if len(gt) == 0:
        return dict(recall=np.nan, precision=np.nan, lead_med=np.nan,
                    n_events=len(ev), n_gt=0, detected=0)
    detected, leads = 0, []
    for g in gt:
        cand = ev[(ev >= g - lead_max) & (ev <= g + tol)]
        if len(cand):
            detected += 1
            leads.append(g - cand[0])       # evento mais cedo da janela
    matched = 0
    for e in ev:
        if np.any((gt >= e - tol) & (gt <= e + lead_max)):
            matched += 1
    return dict(
        recall=detected / len(gt),
        precision=(matched / len(ev)) if len(ev) else np.nan,
        lead_med=float(np.median(leads)) if leads else np.nan,
        n_events=len(ev), n_gt=len(gt), detected=detected,
    )


def direction_hitrate(states, ret: np.ndarray, horizon: int = 20,
                      only_confirmed: bool = True) -> dict:
    """Hit-rate do sinal direcional vs. sinal do retorno forward (horizonte h).

    Mede E[r_{t+h}] · direção > 0. Reporta hit-rate simples e ponderado pela
    confiança |direção|, mais o retorno médio assinado capturado.
    """
    ret = np.asarray(ret)
    hits = []
    weights = []
    signed_ret = []
    for st in states:
        if only_confirmed and st.state != "CONFIRMED":
            continue
        if st.direction == 0.0:
            continue
        t = st.t
        if t + horizon >= len(ret):
            continue
        fwd = float(ret[t:t + horizon].sum())
        s_dir = np.sign(st.direction)
        hits.append(1.0 if s_dir * fwd > 0 else 0.0)
        weights.append(abs(st.direction))
        signed_ret.append(s_dir * fwd)
    if not hits:
        return dict(n=0, hit=np.nan, hit_w=np.nan, mean_signed_ret=np.nan)
    hits = np.array(hits); w = np.array(weights); sr_ = np.array(signed_ret)
    return dict(
        n=len(hits),
        hit=float(hits.mean()),
        hit_w=float((hits * w).sum() / w.sum()),
        mean_signed_ret=float(sr_.mean()),
    )


def _trade_pnls(states, ret: np.ndarray, horizon: int, latency: int,
                cost_bps: float, conf_min: float, size: str):
    """Extrai (pnls, gross) por trade. Uma posição por vez, causal, sem look-ahead."""
    ret = np.asarray(ret)
    n = len(ret)
    cost = cost_bps * 1e-4
    free_at = -1
    pnls, gross_l, entries = [], [], []
    for st in states:
        if st.state != "CONFIRMED" or st.direction == 0.0:
            continue
        if abs(st.direction) < conf_min:
            continue
        entry = st.t - 1 + latency                 # ret[entry:] é o 1º retorno capturado
        if entry < 0 or entry + horizon >= n or entry < free_at:
            continue
        sgn = np.sign(st.direction)
        w = abs(st.direction) if size == "conf" else 1.0
        gross = sgn * float(ret[entry:entry + horizon].sum())
        pnls.append(w * (gross - cost))
        gross_l.append(gross)
        entries.append(entry)                      # índice no slice (eixo temporal comum)
        free_at = entry + horizon
    return np.array(pnls), np.array(gross_l), np.array(entries, dtype=int)


def _bt_metrics(pnls: np.ndarray, gross: np.ndarray) -> dict:
    """Métricas de um vetor de PnL por trade, com t-stat da média (significância)."""
    if len(pnls) == 0:
        return dict(n=0)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    pos, neg = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    sd = pnls.std()
    return dict(
        n=len(pnls),
        hit=float(np.mean(gross > 0)),
        net_mean=float(pnls.mean()),
        net_total=float(cum[-1]),
        sharpe=float(pnls.mean() / sd) if sd > 0 else np.nan,
        tstat=float(pnls.mean() / (sd / np.sqrt(len(pnls)))) if sd > 0 else np.nan,
        profit_factor=float(pos / neg) if neg > 0 else np.inf,
        max_dd=dd,
    )


def block_bootstrap(pnls: np.ndarray, entries: np.ndarray, *, block: int = 390,
                    n_boot: int = 10000, seed: int = 0) -> dict:
    """t-stat honesto via cluster bootstrap por tempo.

    Trades cross-asset no mesmo período são correlacionados (beta de mercado), então
    o t-stat ingênuo (assume independência) é inflado. Agrupa os trades em blocos
    temporais de largura `block` (≥ horizonte; ex.: ~1 dia = 390 barras M1), reamostra
    os blocos INTEIROS com reposição (preserva a co-movimentação intra-bloco) e mede
    o desvio da média ⇒ SE robusto. Reporta o t honesto e o IC 95%.
    """
    rng = np.random.default_rng(seed)
    bins = entries // block
    uniq = np.unique(bins)
    clusters = [pnls[bins == b] for b in uniq]
    sizes = np.array([len(c) for c in clusters])
    n_cl = len(clusters)
    obs = float(pnls.mean())

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_cl, n_cl)          # reamostra blocos com reposição
        tot = sizes[idx].sum()
        boot_means[i] = sum(clusters[j].sum() for j in idx) / tot if tot else np.nan
    se_boot = float(np.nanstd(boot_means))
    se_naive = float(pnls.std() / np.sqrt(len(pnls)))
    lo, hi = np.nanpercentile(boot_means, [2.5, 97.5])
    return dict(
        n=len(pnls), n_clusters=n_cl, mean=obs,
        t_naive=obs / se_naive if se_naive > 0 else np.nan,
        t_honest=obs / se_boot if se_boot > 0 else np.nan,
        se_ratio=se_boot / se_naive if se_naive > 0 else np.nan,
        ci95=(float(lo), float(hi)),
        p_pos=float(np.mean(boot_means <= 0)),      # ~p-valor unilateral (mean>0)
    )


def backtest(states, ret: np.ndarray, *, horizon: int = 60, latency: int = 1,
             cost_bps: float = 1.0, conf_min: float = 0.0,
             size: str = "fixed") -> dict:
    """Backtest causal dos eventos CONFIRMED. Uma posição por vez (flat entre trades).

    Para cada evento com |direction| ≥ conf_min: entra `latency` barras após o sinal
    (modela atraso alarme→execução), segura `horizon` barras, sai. PnL em log-retorno
    menos `cost_bps` round-trip. `size`: 'fixed' (unitário) ou 'conf' (∝ |direction|).
    """
    pn, gr, _ = _trade_pnls(states, ret, horizon, latency, cost_bps, conf_min, size)
    return _bt_metrics(pn, gr)


def portfolio_backtest(symbols, tf: str = "M1", *, lo: int = 0, hi: int = 100_000,
                       embed: int = 70, p: int = 8, alpha: float = 0.02,
                       h: float = 15.0, nu: float = 1.0,
                       direction_mode: str = "reversion",
                       direction_source: str = "drift", conf_gate: str = "entropy",
                       horizon: int = 60, latency: int = 1, cost_bps: float = 1.0,
                       conf_min: float = 0.2, size: str = "fixed",
                       clip: float = 0.5, verbose: bool = True) -> dict:
    """Roda o detector em vários ativos no slice [lo,hi) e AGRUPA os trades.

    Pool de trades entre ativos dá poder estatístico (t-stat). `clip` protege contra
    splits não ajustados. Devolve métricas pooled + tabela por ativo.
    """
    import decomp_pca
    all_pnls, all_gross, all_entries, rows = [], [], [], []
    for sym in symbols:
        try:
            df = decomp_pca.load_ohlcv(sym, tf)
        except Exception:
            continue
        _, ret = decomp_pca.compute_log_returns(df)
        seg = np.clip(ret[lo:hi], -clip, clip)
        if len(seg) < 6 * embed:
            continue
        st = replay(seg, embed=embed, p=p, alpha=alpha, h=h, nu=nu,
                    direction_mode=direction_mode, direction_source=direction_source,
                    conf_gate=conf_gate, track_rotation=False)
        pnls, gross, entries = _trade_pnls(st, seg, horizon, latency, cost_bps,
                                           conf_min, size)
        if len(pnls):
            all_pnls.append(pnls); all_gross.append(gross); all_entries.append(entries)
            m = _bt_metrics(pnls, gross)
            rows.append((sym, m["n"], m["hit"], m["net_total"], m["sharpe"]))
            if verbose:
                print(f"  {sym:>6}: n={m['n']:>3} hit={m['hit']:.3f} "
                      f"net={m['net_total']:>6.3f} Sh/tr={m['sharpe']:>6.3f}")
    if not all_pnls:
        return dict(n=0, per_symbol=[])
    pnls_raw = np.concatenate(all_pnls)
    pooled = _bt_metrics(pnls_raw, np.concatenate(all_gross))
    pooled["per_symbol"] = rows
    pooled["n_symbols"] = len(rows)
    pooled["pnls"] = pnls_raw                       # p/ block bootstrap
    pooled["entries"] = np.concatenate(all_entries)
    return pooled


def walkforward(symbols, tf: str = "M1", *, n_folds: int = 4, fold_len: int = 50_000,
                conf_grid=(0.1, 0.2, 0.3), mode_grid=("reversion", "momentum"),
                embed: int = 70, alpha: float = 0.02, h: float = 15.0, nu: float = 1.0,
                direction_source: str = "ssa", horizon: int = 60, latency: int = 1,
                cost_bps: float = 1.0, clip: float = 0.5) -> dict:
    """Walk-forward com seleção de config 100% OOS, sem look-ahead.

    Em cada fold escolhe a config (direction_mode × conf_min) que maximiza o t-stat
    pooled no fold de CALIBRAÇÃO e aplica no fold de TESTE seguinte. Pool dos trades
    OOS de todos os folds; devolve trades p/ block bootstrap. Janelas temporais
    disjuntas e consecutivas, mesmo universo.
    """
    import decomp_pca
    series = {}
    for sym in symbols:
        try:
            df = decomp_pca.load_ohlcv(sym, tf)
        except Exception:
            continue
        _, ret = decomp_pca.compute_log_returns(df)
        series[sym] = np.clip(ret, -clip, clip)

    _cache = {}
    def states_for(sym, lo, hi, mode):           # replay 1× por (sym, janela, modo)
        key = (sym, lo, hi, mode)
        if key not in _cache:
            seg = series[sym][lo:hi]
            if len(seg) < 6 * embed:
                _cache[key] = (None, seg)
            else:
                _cache[key] = (replay(seg, embed=embed, p=8, alpha=alpha, h=h, nu=nu,
                                      direction_mode=mode,
                                      direction_source=direction_source,
                                      track_rotation=False), seg)
        return _cache[key]

    def pooled(syms, lo, hi, mode, conf):
        ps, gs, es = [], [], []
        for sym in syms:
            st, seg = states_for(sym, lo, hi, mode)
            if st is None:
                continue
            pn, gr, en = _trade_pnls(st, seg, horizon, latency, cost_bps, conf, "fixed")
            if len(pn):
                ps.append(pn); gs.append(gr); es.append(en)
        if not ps:
            return np.array([]), np.array([]), np.array([], dtype=int)
        return np.concatenate(ps), np.concatenate(gs), np.concatenate(es)

    oos_pnls, oos_gross, oos_entries, fold_rows = [], [], [], []
    syms = list(series.keys())
    for k in range(n_folds - 1):
        cal_lo, cal_hi = k * fold_len, (k + 1) * fold_len
        test_lo, test_hi = (k + 1) * fold_len, (k + 2) * fold_len
        # seleciona (modo, conf) que maximiza o t-stat no fold de calibração
        best, best_t = (mode_grid[0], conf_grid[0]), -np.inf
        for mode in mode_grid:
            for c in conf_grid:
                pn, gr, _ = pooled(syms, cal_lo, cal_hi, mode, c)
                t = _bt_metrics(pn, gr).get("tstat", -np.inf) if len(pn) else -np.inf
                if np.isfinite(t) and t > best_t:
                    best_t, best = t, (mode, c)
        # aplica no fold de teste (OOS) — globalmente alinhado por offset test_lo
        bmode, bconf = best
        pn, gr, en = pooled(syms, test_lo, test_hi, bmode, bconf)
        m = _bt_metrics(pn, gr)
        fold_rows.append((k, bmode, bconf, m.get("n", 0), m.get("hit", np.nan),
                          m.get("sharpe", np.nan), m.get("tstat", np.nan)))
        if len(pn):
            oos_pnls.append(pn); oos_gross.append(gr)
            oos_entries.append(en + test_lo)     # índice global p/ blocos do bootstrap
    if not oos_pnls:
        return dict(n=0, folds=fold_rows)
    pn_all = np.concatenate(oos_pnls)
    out = _bt_metrics(pn_all, np.concatenate(oos_gross))
    out["folds"] = fold_rows
    out["pnls"] = pn_all
    out["entries"] = np.concatenate(oos_entries)
    return out


def plot_states(states: list[RegimeState], price: np.ndarray,
                path: str, title: str = ""):
    """Trajetória dos sinais espectrais com eventos e estados sombreados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ti = np.array([st.t for st in states])
    px = price[ti - 1]
    trace = np.array([st.trace for st in states])
    lamp  = np.array([st.lam_plus for st in states])
    lam1  = np.array([st.lam1 for st in states])
    m     = np.array([st.m for st in states])
    S     = np.array([st.entropy for st in states])
    nov   = np.array([st.novelty for st in states])
    ang   = np.array([st.theta for st in states])
    watch = np.array([st.state == "WATCH" for st in states])
    conf  = np.array([st.state == "CONFIRMED" for st in states])
    evs   = confirmed_events(states)

    fig, ax = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(title, fontsize=11)

    def shade(a):
        a.fill_between(ti, *a.get_ylim(), where=watch, color="#F0A500",
                       alpha=0.12, step="mid", lw=0)
        a.fill_between(ti, *a.get_ylim(), where=conf, color="#D7263D",
                       alpha=0.18, step="mid", lw=0)
        for e in evs:
            a.axvline(e.t, color="#D7263D", lw=0.4, alpha=0.5)

    ax[0].plot(ti, px, color="#222", lw=0.7); ax[0].set_ylabel("preco")
    # direção dos eventos: ^ verde = long, v vermelho = short (tamanho ~ confiança)
    for e in evs:
        if e.direction == 0:
            continue
        up = e.direction > 0
        ax[0].scatter(e.t, price[e.t - 1], marker="^" if up else "v",
                      s=20 + 120 * abs(e.direction),
                      color="#2ca02c" if up else "#D7263D", zorder=5, alpha=0.8)
    ax[1].semilogy(ti, trace, color="#1f77b4", lw=0.7, label="trace (vol^2*d)")
    ax[1].semilogy(ti, lam1,  color="#2ca02c", lw=0.5, alpha=0.7, label="lambda_1")
    ax[1].semilogy(ti, lamp,  color="#999", lw=0.5, ls="--", label="lambda_+ (MP)")
    ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_ylabel("autovalores")
    ax[2].plot(ti, m, color="#9467bd", lw=0.7); ax[2].set_ylabel("m (modos > MP)")
    ax[3].plot(ti, S, color="#8c564b", lw=0.7); ax[3].set_ylabel("entropia S")
    ax[3].set_ylim(0, 1.02)
    ax[4].plot(ti, nov, color="#e377c2", lw=0.6, label="novelty")
    ax4b = ax[4].twinx()
    ax4b.plot(ti, ang, color="#17becf", lw=0.5, alpha=0.6, label="angulo subesp.")
    ax[4].set_ylabel("novelty"); ax4b.set_ylabel("angulo (rad)")
    ax[4].set_yscale("log"); ax[4].set_xlabel("tick")

    for a in ax:
        shade(a)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _main():
    import decomp_pca   # carregador Parquet do próprio projeto

    ap = argparse.ArgumentParser(description="Replay do detector de regime streaming")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", default="M1")
    ap.add_argument("--embed", type=int, default=70)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--h", type=float, default=15.0)
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = série inteira")
    ap.add_argument("--dir-mode", default="reversion",
                    choices=["auto", "momentum", "reversion"])
    ap.add_argument("--horizon", type=int, default=20, help="horizonte do hit-rate")
    ap.add_argument("--plot", default="", help="caminho do PNG (vazio = sem plot)")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--conf-min", type=float, default=0.2)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--latency", type=int, default=1)
    args = ap.parse_args()

    df = decomp_pca.load_ohlcv(args.symbol, args.tf)
    ts, ret_C = decomp_pca.compute_log_returns(df)
    close = df["close"].values[1:].astype(np.float64)   # alinhado a ret_C
    if args.limit:
        ts, ret_C, close = ts[:args.limit], ret_C[:args.limit], close[:args.limit]
    print(f"{args.symbol} {args.tf}: {len(ret_C)} retornos | embed={args.embed} "
          f"alpha={args.alpha} (n_eff~{(2-args.alpha)/args.alpha:.0f})")

    states = replay(ret_C, embed=args.embed, p=args.p, alpha=args.alpha,
                    h=args.h, nu=args.nu, direction_mode=args.dir_mode)
    confirmed = confirmed_events(states)

    hr = direction_hitrate(states, ret_C, horizon=args.horizon)
    print(f"direção ({args.dir_mode}, h={args.horizon}): n={hr['n']} "
          f"hit={hr['hit']:.3f} hit_ponderado={hr['hit_w']:.3f}")

    if args.backtest:
        b = backtest(states, ret_C, horizon=args.horizon, latency=args.latency,
                     cost_bps=args.cost_bps, conf_min=args.conf_min)
        if b["n"]:
            print(f"backtest (conf_min={args.conf_min} lat={args.latency} "
                  f"cost={args.cost_bps}bp hold={args.horizon}): n={b['n']} "
                  f"hit={b['hit']:.3f} net_tot={b['net_total']:.3f} "
                  f"sharpe/tr={b['sharpe']:.3f} PF={b['profit_factor']:.2f} "
                  f"maxDD={b['max_dd']:.3f}")
        else:
            print("backtest: sem trades (conf_min alto demais?)")

    if args.plot:
        plot_states(states, close, args.plot,
                    title=f"{args.symbol} {args.tf} | embed={args.embed} "
                          f"alpha={args.alpha} h={args.h} nu={args.nu} "
                          f"| {len(confirmed)} eventos")
        print(f"plot salvo em {args.plot}")

    print(f"\n{len(confirmed)} eventos CONFIRMED:")
    for st in confirmed[:50]:
        when = ts[st.t - 1] if st.t - 1 < len(ts) else "?"
        arrow = "^" if st.direction > 0 else ("v" if st.direction < 0 else "-")
        print(f"  t={st.t:>7} {str(when)[:19]} | m={st.m} "
              f"trace={st.trace:.2e} S={st.entropy:.3f} ang={st.theta:.3f} "
              f"dir={arrow}{abs(st.direction):.2f} alarms={st.alarms}")


if __name__ == "__main__":
    _main()
