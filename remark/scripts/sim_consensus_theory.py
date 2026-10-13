"""CPU simulation that checks the theoretical claims on consensus rewards (numpy only; ~2 threads; minutes).

World. M items with Zipf-like global popularity; clusters ("genres") perturb the item logits; each thread perturbs
its cluster's logits. A responder recommends 1 + Poisson(lam) items (capped) drawn without replacement from the
thread's softmax (ordered Gumbel top-m). p_i(x) = P(i in S | x) is estimated from n_mc responders per eval thread.
Responders per thread follow the Reddit-v2 train histogram (remark/scripts/crs_responder_stats.py). The policy is
Plackett-Luce over items (K items without replacement) with per-thread logits beta * cluster logits + noise: it
knows the genre, not the thread. The estimator is Rank-GRPO's per-position one:
    A[g,k] = (R[g,k] - mean_g R[.,k]) / sigma_k,   g_hat = (1/G) sum_{g,k} A[g,k] d/dtheta log pi(L_k | L_<k),
with sigma_k = 1 (none), the group std + eps (group) or the batch std + eps (batch; std with ddof=1 as in torch).
Reference gradients h(r) = E sum_k r(L_k) d log pi(L_k | L_<k) are linear in the item reward vector r and are
computed once per eval thread by Monte Carlo (h(r) = r @ V).

A  reward choice (sigma = 1), n responders per eval thread, n in n_list: single vs consensus vs shrinkage
   (alpha chosen by leave-one-responder-out Brier on the world data). Cosine of the batch gradient to h_pop,
   per-step SNR at fixed data, exact decomposition Var(g_S) = Var(g_C) + E Var(g_S | lists), zero-signal groups,
   cosine of the expected gradient to h_pop over an alpha grid, and MSE of p_tilde vs p_hat.
B  normalization with Reddit-like n: group vs batch std for single / consensus / shrinkage rewards: nonzero group
   std, amplification rho = (sigma_batch + eps)/(s_group + eps), squared-advantage mass on groups with s below the
   binary minimum 1/sqrt(G), advantage reliability across two independent responder samples, gradient cosine to h_pop.
C  exact nonzero group-std range for binary rewards (Proposition 4c).

Outputs: remark/outputs/theory/{sim_results.json, manifest.json, fig_reward_variance.pdf, fig_adv_norm.pdf}.
Usage: env/bin/python remark/scripts/sim_consensus_theory.py [--quick]
(matplotlib in this env needs pyparsing on PYTHONPATH; without matplotlib the figures are skipped.)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "remark/configs/sim_consensus_theory.yaml"
OUT = REPO / "remark/outputs/theory"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


# ----------------------------------------------------------------------------------------------------------- world
class World:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        w, pol, pr = cfg["world"], cfg["policy"], cfg["prior"]
        M, C, Tc = w["n_items"], w["n_clusters"], w["threads_per_cluster"]
        self.M, self.T = M, C * Tc
        glob = -w["zipf_s"] * np.log(rng.permutation(M) + 1.0)
        self.clus = glob[None] + w["sigma_cluster"] * rng.standard_normal((C, M))
        self.cl = np.repeat(np.arange(C), Tc)
        self.logits = self.clus[self.cl] + w["sigma_thread"] * rng.standard_normal((self.T, M))
        self.pol = pol["beta_cluster"] * self.clus[self.cl] + pol["sigma_noise"] * rng.standard_normal((self.T, M))
        self.lam, self.mmax = w["items_per_responder_lam"], w["items_per_responder_max"]
        k, err = pr["k_neighbors"], pr["neighbor_error"]
        self.nb = np.zeros((self.T, k), dtype=int)
        for x in range(self.T):
            same = np.flatnonzero((self.cl == self.cl[x]) & (np.arange(self.T) != x))
            other = np.flatnonzero(np.arange(self.T) != x)
            picks: list[int] = []
            while len(picks) < k:
                c = int(rng.choice(other if rng.random() < err else same))
                if c not in picks:
                    picks.append(c)
            self.nb[x] = picks
        self.global_mix = pr["global_mix"]
        hist_path = REPO / w["n_hist_file"]
        if hist_path.exists():
            h = np.array(json.loads(hist_path.read_text())["n_hist"], dtype=float)
            self.n_vals, self.n_prob, self.n_source = h[:, 0].astype(int), h[:, 1] / h[:, 1].sum(), str(hist_path)
        else:  # fallback: truncated discrete power law with a similar median/mean
            v = np.arange(1, 401)
            pmf = v ** -1.9
            self.n_vals, self.n_prob, self.n_source = v, pmf / pmf.sum(), "fallback power law n^-1.9, n<=400"

    def sample_n(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(self.n_vals, size=size, p=self.n_prob)

    def responders(self, x: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, M) 0/1 item indicators; each responder names m items drawn without replacement."""
        m = np.minimum(1 + rng.poisson(self.lam, size=n), self.mmax)
        g = self.logits[x][None] + rng.gumbel(size=(n, self.M))
        top = np.argpartition(-g, self.mmax - 1, axis=1)[:, : self.mmax]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(g, top, 1), axis=1), 1)
        keep = np.arange(self.mmax)[None] < m[:, None]
        Z = np.zeros((n, self.M))
        Z[np.repeat(np.arange(n), self.mmax).reshape(n, -1)[keep], top[keep]] = 1.0
        return Z

    def draw_data(self, rng: np.random.Generator) -> dict:
        """One training set: Reddit-like n per thread; neighbour prior mu for every thread (self excluded)."""
        n = self.sample_n(rng, self.T)
        Zs = [self.responders(x, int(n[x]), rng) for x in range(self.T)]
        c = np.stack([Z.sum(0) for Z in Zs])
        phat = c / n[:, None]
        glob = c.sum(0) / n.sum()
        mu = (1 - self.global_mix) * phat[self.nb].mean(1) + self.global_mix * glob[None]
        return {"n": n, "Zs": Zs, "c": c, "mu": mu}


# ------------------------------------------------------------------------------------------ Plackett-Luce policy
def pl_sample(logits: np.ndarray, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """logits (X, M) -> (X, n, k) ordered samples without replacement (Gumbel top-k)."""
    g = logits[:, None, :] + rng.gumbel(size=(logits.shape[0], n, logits.shape[1]))
    idx = np.argpartition(-g, k - 1, axis=-1)[..., :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(g, idx, -1), axis=-1), -1)


def pl_scores(logits: np.ndarray, L: np.ndarray) -> np.ndarray:
    """d/dlogits log pi(L_k | L_<k) for a Plackett-Luce policy; (X, M), (X, N, K) -> (X, N, K, M)."""
    X, N, K = L.shape
    pi = softmax(logits)
    oh = np.zeros((X, N, K, logits.shape[1]))
    np.put_along_axis(oh, L[..., None], 1.0, axis=-1)
    pL = pi[np.arange(X)[:, None, None], L]
    rem = 1.0 - (np.cumsum(pL, -1) - pL)
    avail = 1.0 - (np.cumsum(oh, axis=2) - oh)
    return oh - pi[:, None, None, :] * avail / rem[..., None]


def reference_V(logits: np.ndarray, n_lists: int, K: int, rng: np.random.Generator, chunk: int = 500):
    """V (M, M) with h(r) = r @ V ~= E sum_k r(L_k) score_k (scores centred per position); plus two half estimates."""
    M = logits.shape[0]
    chunk = max(1, min(chunk, n_lists // 2))
    acc = [dict(V=np.zeros((M, M)), F=np.zeros((K, M)), S=np.zeros((K, M)), N=0) for _ in range(2)]
    for j in range(0, n_lists, chunk):
        a = acc[0] if j < n_lists // 2 else acc[1]
        L = pl_sample(logits[None], chunk, K, rng)
        S = pl_scores(logits[None], L)[0]  # (chunk, K, M)
        Lf, Sf = L[0].ravel(), S.reshape(-1, M)
        order = np.argsort(Lf, kind="stable")
        Ls = Lf[order]
        starts = np.flatnonzero(np.r_[True, Ls[1:] != Ls[:-1]])
        a["V"][Ls[starts]] += np.add.reduceat(Sf[order], starts, axis=0)
        np.add.at(a["F"], (np.tile(np.arange(K), chunk), Lf), 1.0)
        a["S"] += S.sum(0)
        a["N"] += chunk

    def finish(parts):
        N = sum(p["N"] for p in parts)
        V = sum(p["V"] for p in parts)
        F = sum(p["F"] for p in parts)
        Sbar = sum(p["S"] for p in parts) / N
        return (V - F.T @ Sbar) / N

    return finish(acc), finish([acc[0]]), finish([acc[1]])


def advantages(R: np.ndarray, norm: str, eps: float) -> np.ndarray:
    """R (X, G, K) per-position rewards -> advantages with group-mean baseline and the given std normalization."""
    A = R - R.mean(1, keepdims=True)
    if norm == "group":
        return A / (R.std(1, ddof=1, keepdims=True) + eps)
    if norm == "batch":
        return A / (R.reshape(-1, R.shape[-1]).std(0, ddof=1)[None, None] + eps)
    return A


# ------------------------------------------------------------------------------------------ empirical Bayes alpha
def loro_brier(Zs, mus, alphas, weights=None) -> np.ndarray:
    """Leave-one-responder-out Brier score per alpha, averaged over threads with n >= 2 (thread-uniform).
    weights (T, M): optional per-item weights (policy exposure P(i in L | x)); None = uniform over the catalog."""
    al = np.asarray(alphas, dtype=float)[:, None, None]
    out, cnt = np.zeros(len(alphas)), 0
    for j, (Z, mu) in enumerate(zip(Zs, mus)):
        n = Z.shape[0]
        if n < 2:
            continue
        c = Z.sum(0)
        w = np.ones(Z.shape[1]) if weights is None else weights[j] / weights[j].mean()
        pt = (c[None, None] - Z[None] + al * mu[None, None]) / (n - 1 + al)
        out += (w[None, None] * (Z[None] - pt) ** 2).sum(-1).mean(-1)
        cnt += 1
    return out / max(cnt, 1)


def loro_advantage(Zs, mus, lists, alphas) -> np.ndarray:
    """Leave-one-responder-out advantage alignment per alpha (threads with n >= 2).
    lists[j]: (R, G, K) policy lists for thread j. For held-out responder r, A_alpha = group-centred per-position
    rewards under p_tilde^(-r)(alpha), A_r = group-centred hits of responder r on the same lists. Since A is linear in
    the item rewards and responder r is independent of the others, E[<A_alpha, A_r>] = <A_alpha, A(p)>. Score =
    sum <A_alpha, A_r> / sqrt(sum ||A_alpha||^2): a scale-free (batch-normalized) alignment with the population
    advantage; the alpha-free factor ||A(p)|| is dropped."""
    num, den = np.zeros(len(alphas)), np.zeros(len(alphas))
    for Z, mu, L in zip(Zs, mus, lists):
        n = Z.shape[0]
        if n < 2:
            continue
        Rg, G, K = L.shape
        idx = L.reshape(-1)
        Ar = Z[:, idx].reshape(n, Rg, G, K)
        Ar = Ar - Ar.mean(2, keepdims=True)
        c = Z.sum(0)
        for j, al in enumerate(alphas):
            pt = (c[None] - Z + al * mu[None]) / (n - 1 + al)  # (n, M)
            Aa = pt[:, idx].reshape(n, Rg, G, K)
            Aa = Aa - Aa.mean(2, keepdims=True)
            num[j] += float((Aa * Ar).sum())
            den[j] += float((Aa ** 2).sum())
    return num / np.sqrt(den + 1e-300)


def exposure(logits: np.ndarray, K: int, n_lists: int, rng: np.random.Generator, chunk: int = 40) -> np.ndarray:
    """P(i in L | x) under the Plackett-Luce policy, (T, M), by Monte Carlo."""
    out = np.zeros_like(logits)
    for a in range(0, logits.shape[0], chunk):
        L = pl_sample(logits[a:a + chunk], n_lists, K, rng)
        for j in range(L.shape[0]):
            out[a + j] = np.bincount(L[j].ravel(), minlength=logits.shape[1]) / n_lists
    return out


def log_betabin_alpha_terms(c: np.ndarray, n: int, a: np.ndarray, b: np.ndarray) -> float:
    """sum_i [log B(c_i + a_i, n - c_i + b_i) - log B(a_i, b_i)] via rising factorials (binomial term omitted)."""
    t = np.arange(n)
    la = np.concatenate([np.zeros((len(a), 1)), np.cumsum(np.log(a[:, None] + t[None]), 1)], 1)
    lb = np.concatenate([np.zeros((len(b), 1)), np.cumsum(np.log(b[:, None] + t[None]), 1)], 1)
    lab = np.log((a + b)[:, None] + t[None]).sum(1)
    idx = np.arange(len(a))
    ci = c.astype(int)
    return float((la[idx, ci] + lb[idx, n - ci] - lab).sum())


def marginal_loglik(Zs, mus, alphas) -> np.ndarray:
    """Beta-Binomial marginal log-likelihood per alpha (> 0) over threads with n >= 2 (n = 1 carries no alpha info)."""
    out = np.zeros(len(alphas))
    for Z, mu in zip(Zs, mus):
        n = Z.shape[0]
        if n < 2:
            continue
        c = Z.sum(0)
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        for j, al in enumerate(alphas):
            out[j] += log_betabin_alpha_terms(c, n, al * mu, al * (1 - mu)) if al > 0 else -np.inf
    return out


# ------------------------------------------------------------------------------------------------------ experiments
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for debugging")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from sim_results.json")
    args = ap.parse_args()
    if args.figures_only:
        r = json.loads((OUT / "sim_results.json").read_text())
        make_figures(r["exp_a"], r["exp_b"], r["exp_c"]["G"], "")
        return
    cfg = yaml.safe_load(CFG.read_text())
    if args.quick:
        cfg["exp_a"].update(data_draws=2, reps=8, n_list=[1, 4, 16])
        cfg["exp_b"]["reps"] = 20
        cfg["exp_d"].update(sigma_thread=[0.5], n_list=[1, 4], data_draws=1, n_ref_lists=300)
        cfg["estimator"]["n_ref_lists"] = 1000
    t0 = time.time()
    rng = np.random.default_rng(cfg["seed"])
    est = cfg["estimator"]
    K, G, B, eps = est["list_len"], est["group_size"], est["batch_threads"], est["eps"]
    alphas = [float(a) for a in cfg["prior"]["alpha_grid"]]
    world = World(cfg, rng)
    M = world.M
    ev = rng.choice(world.T, B, replace=False)
    pol = world.pol[ev]
    p_true = np.stack([world.responders(int(x), cfg["world"]["n_mc_true_p"], rng).mean(0) for x in ev])

    Vs = [reference_V(pol[b], est["n_ref_lists"], K, rng) for b in range(B)]
    V = np.stack([v[0] for v in Vs])

    def h(r: np.ndarray, VV: np.ndarray = V) -> np.ndarray:
        return np.einsum("bm,bmd->bd", r, VV)

    h_pop = h(p_true)
    mc_check = cos(h(p_true, np.stack([v[1] for v in Vs])), h(p_true, np.stack([v[2] for v in Vs])))
    del Vs
    t_ref = time.time() - t0
    print(f"reference gradients: {t_ref:.1f}s, split-half cosine {mc_check:.4f}", flush=True)

    # training-set draws: neighbour priors and empirical-Bayes alpha
    D = cfg["exp_a"]["data_draws"]
    expo = exposure(world.pol, K, 200, rng)
    pool_lists = [pl_sample(world.pol[x][None], 2 * G, K, rng)[0].reshape(2, G, K) for x in range(world.T)]
    worlds, eb = [], []
    for d in range(D):
        wd = world.draw_data(rng)
        brier = loro_brier(wd["Zs"], wd["mu"], alphas)
        brier_x = loro_brier(wd["Zs"], wd["mu"], alphas, weights=expo)
        mll = marginal_loglik(wd["Zs"], wd["mu"], alphas)
        adv = loro_advantage(wd["Zs"], wd["mu"], pool_lists, alphas)
        eb.append({"alpha_loro": alphas[int(np.argmin(brier))], "alpha_loro_exposure": alphas[int(np.argmin(brier_x))],
                   "alpha_loro_adv": alphas[int(np.argmax(adv))], "loro_adv": adv.tolist(),
                   "alpha_ml": alphas[int(np.argmax(mll))], "loro_brier": brier.tolist(),
                   "loro_brier_exposure": brier_x.tolist(), "marginal_loglik": [float(v) for v in mll]})
        worlds.append(wd)
    print(f"EB alpha LORO-adv {[e['alpha_loro_adv'] for e in eb]}; LORO-Brier uniform {[e['alpha_loro'] for e in eb]}; "
          f"LORO-Brier exposure "
          f"{[e['alpha_loro_exposure'] for e in eb]}; ML {[e['alpha_ml'] for e in eb]}", flush=True)

    def lists_and_scores():
        L = pl_sample(pol, G, K, rng)
        return L, pl_scores(pol, L)

    # ---------------------------------------------------------------- A: reward choice, sigma = 1
    res_a = []
    R_a = cfg["exp_a"]["reps"]
    c_exp = (G - 1) / G  # E g_hat = (G-1)/G h(r) for the group-mean baseline that includes the sample itself
    rb_err = 0.0
    for n in cfg["exp_a"]["n_list"]:
        rec: dict = {"n": n, "draws": []}
        for d in range(D):
            wd, aEB, aU = worlds[d], eb[d]["alpha_loro_adv"], eb[d]["alpha_loro"]
            mu = wd["mu"][ev]
            Z = np.stack([world.responders(int(x), n, rng) for x in ev])  # (B, n, M)
            cnt = Z.sum(1)
            phat = cnt / n

            def ptil(a: float) -> np.ndarray:
                return (cnt + a * mu) / (n + a)

            h_emp, h_eb = h(phat), h(ptil(aEB))
            dr = {"alpha_eb": aEB, "alpha_u": aU,
                  "cos_Eg_pop_consensus": cos(h_emp, h_pop),
                  "cos_Eg_pop_by_alpha": [cos(h(ptil(a)), h_pop) for a in alphas],
                  "mse_by_alpha": [float(((ptil(a) - p_true) ** 2).sum(1).mean()) for a in alphas],
                  "mse_eb": float(((ptil(aEB) - p_true) ** 2).sum(1).mean()),
                  "mse_u": float(((ptil(aU) - p_true) ** 2).sum(1).mean()),
                  "mse_plugin": float(((phat - p_true) ** 2).sum(1).mean())}
            if n >= 2:
                dr["alpha_loro_eval"] = alphas[int(np.argmin(loro_brier(list(Z), list(mu), alphas)))]
            cs, cc, ct, cu, cse, cce = [], [], [], [], [], []
            sum_c, sum_t = np.zeros((B, M)), np.zeros((B, M))
            sq_c = sq_t = condvar = 0.0
            zero_s = zero_c = zero_t = 0.0
            for rep in range(R_a):
                L, S = lists_and_scores()
                Lf = L.reshape(B, G * K)
                Sc = (S - S.mean(1, keepdims=True)).reshape(B, G * K, M) / G

                def grad(rv: np.ndarray) -> np.ndarray:
                    return np.einsum("bq,bqm->bm", np.take_along_axis(rv, Lf, 1), Sc)

                g_c, g_t, g_u = grad(phat), grad(ptil(aEB)), grad(ptil(aU))
                R_all = np.take_along_axis(Z, np.repeat(Lf[:, None, :], n, 1), 2)  # (B, n, G*K)
                g_all = np.matmul(R_all, Sc)  # (B, n, M): single-responder gradient for every responder
                if rep == 0:
                    rb_err = max(rb_err, float(np.abs(g_all.mean(1) - g_c).max() / (np.abs(g_c).max() + 1e-300)))
                g_s = g_all[np.arange(B), rng.integers(n, size=B)]
                condvar += float(((g_all - g_c[:, None]) ** 2).sum(-1).mean(1).sum())
                cs.append(cos(g_s, h_pop)); cc.append(cos(g_c, h_pop)); ct.append(cos(g_t, h_pop))
                cu.append(cos(g_u, h_pop))
                cse.append(cos(g_s, h_emp)); cce.append(cos(g_c, h_emp))
                sum_c += g_c; sum_t += g_t
                sq_c += float((g_c ** 2).sum()); sq_t += float((g_t ** 2).sum())
                Rs = R_all.reshape(B, n, G, K)
                zero_s += float((Rs.std(2) == 0).all(-1).mean())
                zero_c += float((np.take_along_axis(phat, Lf, 1).reshape(B, G, K).std(1) == 0).all(-1).mean())
                zero_t += float((np.take_along_axis(ptil(aEB), Lf, 1).reshape(B, G, K).std(1) == 0).all(-1).mean())
            corr_f = R_a / (R_a - 1)
            var_c = (sq_c / R_a - float(((sum_c / R_a) ** 2).sum())) * corr_f
            var_t = (sq_t / R_a - float(((sum_t / R_a) ** 2).sum())) * corr_f
            cond = condvar / R_a
            var_s = var_c + cond
            sig_emp = c_exp ** 2 * float((h_emp ** 2).sum())
            sig_eb = c_exp ** 2 * float((h_eb ** 2).sum())
            dr.update({
                "cos_pop": {"single": float(np.mean(cs)), "consensus": float(np.mean(cc)), "shrink_eb": float(np.mean(ct)),
                            "shrink_u": float(np.mean(cu))},
                "cos_emp": {"single": float(np.mean(cse)), "consensus": float(np.mean(cce))},
                "trvar": {"single": var_s, "consensus": var_c, "shrink_eb": var_t, "E_condvar_single": cond},
                "frac_var_from_responder_sampling": cond / var_s,
                "snr": {"single": sig_emp / var_s, "consensus": sig_emp / var_c, "shrink_eb": sig_eb / var_t},
                "zero_signal_group_rate": {"single": zero_s / R_a, "consensus": zero_c / R_a, "shrink_eb": zero_t / R_a},
                "cos_mean_gc_vs_h_emp": cos(sum_c / R_a, h_emp)})
            rec["draws"].append(dr)
        dd = rec["draws"]

        def avg(key: str, sub: str | None = None) -> float:
            return float(np.mean([x[key][sub] if sub else x[key] for x in dd]))

        def sd(key: str, sub: str | None = None) -> float:
            return float(np.std([x[key][sub] if sub else x[key] for x in dd], ddof=1)) if len(dd) > 1 else 0.0

        by_alpha = np.mean([x["cos_Eg_pop_by_alpha"] for x in dd], 0)
        mse_alpha = np.mean([x["mse_by_alpha"] for x in dd], 0)
        rec["summary"] = {
            **{f"cos_pop_{e}": avg("cos_pop", e) for e in ("single", "consensus", "shrink_eb", "shrink_u")},
            **{f"cos_pop_{e}_sd": sd("cos_pop", e) for e in ("single", "consensus", "shrink_eb", "shrink_u")},
            "alpha_eb_mean": avg("alpha_eb"), "alpha_u_mean": avg("alpha_u"),
            "cos_Eg_pop_consensus": avg("cos_Eg_pop_consensus"),
            "cos_Eg_pop_shrink_eb": float(np.mean([x["cos_Eg_pop_by_alpha"][alphas.index(x["alpha_eb"])] for x in dd])),
            "cos_Eg_pop_shrink_u": float(np.mean([x["cos_Eg_pop_by_alpha"][alphas.index(x["alpha_u"])] for x in dd])),
            "cos_Eg_pop_by_alpha": by_alpha.tolist(), "mse_by_alpha": mse_alpha.tolist(),
            "alpha_best_cos_Eg": alphas[int(np.argmax(by_alpha))], "cos_Eg_pop_best_alpha": float(by_alpha.max()),
            "alpha_best_mse": alphas[int(np.argmin(mse_alpha))],
            **{f"snr_{e}": avg("snr", e) for e in ("single", "consensus", "shrink_eb")},
            "frac_var_from_responder_sampling": avg("frac_var_from_responder_sampling"),
            "var_ratio_single_over_consensus": float(np.mean([x["trvar"]["single"] / x["trvar"]["consensus"] for x in dd])),
            **{f"zero_signal_{e}": avg("zero_signal_group_rate", e) for e in ("single", "consensus", "shrink_eb")},
            "mse_plugin": avg("mse_plugin"), "mse_eb": avg("mse_eb"), "mse_best_alpha": float(mse_alpha.min()),
            "mse_ratio_eb": avg("mse_eb") / avg("mse_plugin"), "mse_ratio_u": avg("mse_u") / avg("mse_plugin"),
            "mse_ratio_best_alpha": float(mse_alpha.min()) / avg("mse_plugin"),
            "cos_mean_gc_vs_h_emp": avg("cos_mean_gc_vs_h_emp"),
            "alpha_loro_eval": (float(np.mean([x["alpha_loro_eval"] for x in dd])) if n >= 2 else None),
        }
        res_a.append(rec)
        print(f"A n={n}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in rec["summary"].items()}), flush=True)

    # ---------------------------------------------------------------- B: normalization, Reddit-like n
    R_b = cfg["exp_b"]["reps"]
    rtypes, norms = ("single", "consensus", "shrink_eb", "shrink_u"), ("none", "group", "batch")
    cosg = {r: {m: [] for m in norms} for r in rtypes}
    rel = {r: {m: [] for m in norms} for r in rtypes}
    grel = {r: {m: [] for m in norms} for r in rtypes}
    s_nonzero = {r: [] for r in rtypes}
    rho = {r: [] for r in rtypes}
    sub_mass = {r: {m: [] for m in ("group", "batch")} for r in rtypes}
    zero_b = {r: [] for r in rtypes}
    sigma_b = {r: [] for r in rtypes}
    n_seen = []
    s_bin_min = 1 / np.sqrt(G)
    for rep in range(R_b):
        nx = world.sample_n(rng, B)
        n_seen.append(nx)
        rv = []
        for di in (rep % D, (rep + 1) % D):  # independent copy: fresh responders and a different prior draw
            wd, a, au = worlds[di], eb[di]["alpha_loro_adv"], eb[di]["alpha_loro"]
            Zs = [world.responders(int(x), int(nx[b]), rng) for b, x in enumerate(ev)]
            cnt = np.stack([Z.sum(0) for Z in Zs])
            rv.append({"single": np.stack([Z[rng.integers(Z.shape[0])] for Z in Zs]),
                       "consensus": cnt / nx[:, None],
                       "shrink_eb": (cnt + a * wd["mu"][ev]) / (nx[:, None] + a),
                       "shrink_u": (cnt + au * wd["mu"][ev]) / (nx[:, None] + au)})
        L, S = lists_and_scores()
        Lf = L.reshape(B, G * K)
        for r in rtypes:
            R1 = np.take_along_axis(rv[0][r], Lf, 1).reshape(B, G, K)
            R2 = np.take_along_axis(rv[1][r], Lf, 1).reshape(B, G, K)
            s = R1.std(1, ddof=1)
            sb = R1.reshape(-1, K).std(0, ddof=1)
            nz = s > 0
            s_nonzero[r].append(s[nz])
            rho[r].append(((sb[None] + eps) / (s + eps))[nz])
            sigma_b[r].append(sb)
            zero_b[r].append(float((~nz).all(-1).mean()))
            for m in norms:
                A1, A2 = advantages(R1, m, eps), advantages(R2, m, eps)
                g1 = np.einsum("bgk,bgkm->bm", A1, S) / G
                g2 = np.einsum("bgk,bgkm->bm", A2, S) / G
                cosg[r][m].append(cos(g1, h_pop))
                rel[r][m].append(corr(A1, A2))
                grel[r][m].append(cos(g1, g2))
                if m in ("group", "batch"):
                    mass = (A1 ** 2).sum(1)  # (B, K)
                    sub_mass[r][m].append(float(mass[nz & (s < s_bin_min - 1e-12)].sum() / (mass.sum() + 1e-300)))
    res_b = {"n_mean_sampled": float(np.mean(n_seen)), "n_median_sampled": float(np.median(n_seen))}
    for r in rtypes:
        s_all, rho_all = np.concatenate(s_nonzero[r]), np.concatenate(rho[r])
        res_b[r] = {
            "zero_signal_group_rate": float(np.mean(zero_b[r])),
            "sigma_batch_mean_over_positions": float(np.mean(sigma_b[r])),
            "s_nonzero_quantiles": {q: float(np.quantile(s_all, float(q))) for q in ("0.01", "0.1", "0.5", "0.9")},
            "frac_nonzero_groups_below_binary_min": float((s_all < s_bin_min - 1e-12).mean()),
            "rho_quantiles": {q: float(np.quantile(rho_all, float(q))) for q in ("0.1", "0.5", "0.9", "0.99")},
            "s_nonzero_max_over_min": float(s_all.max() / s_all.min()),
            "s_nonzero_ecdf_q": np.quantile(s_all, np.linspace(0, 1, 201)).tolist(),
            "sq_adv_mass_on_sub_binary_groups": {m: float(np.mean(sub_mass[r][m])) for m in ("group", "batch")},
            "cos_pop": {m: float(np.mean(cosg[r][m])) for m in norms},
            "cos_pop_sd": {m: float(np.std(cosg[r][m], ddof=1)) for m in norms},
            "adv_reliability": {m: float(np.mean(rel[r][m])) for m in norms},
            "grad_reliability": {m: float(np.mean(grel[r][m])) for m in norms},
        }
        print(f"B {r}: " + json.dumps(res_b[r]), flush=True)

    # ---------------------------------------------------------------- D: prior informativeness (expected gradients)
    res_d = []
    for st in cfg["exp_d"]["sigma_thread"]:
        cfg_d = json.loads(json.dumps(cfg))
        cfg_d["world"]["sigma_thread"] = st
        rng_d = np.random.default_rng(cfg["seed"] + 1000 + int(10 * st))
        wdd = World(cfg_d, rng_d)
        ev_d = rng_d.choice(wdd.T, B, replace=False)
        p_d = np.stack([wdd.responders(int(x), cfg["world"]["n_mc_true_p"], rng_d).mean(0) for x in ev_d])
        V_d = np.stack([reference_V(wdd.pol[x], cfg["exp_d"]["n_ref_lists"], K, rng_d)[0] for x in ev_d])
        hp = h(p_d, V_d)
        pl_d = [pl_sample(wdd.pol[x][None], 2 * G, K, rng_d)[0].reshape(2, G, K) for x in range(wdd.T)]
        rec_d = {"sigma_thread": st, "by_n": {}}
        a_adv, a_bri = [], []
        curves = {n: [] for n in cfg["exp_d"]["n_list"]}
        for _ in range(cfg["exp_d"]["data_draws"]):
            wdat = wdd.draw_data(rng_d)
            a_adv.append(alphas[int(np.argmax(loro_advantage(wdat["Zs"], wdat["mu"], pl_d, alphas)))])
            a_bri.append(alphas[int(np.argmin(loro_brier(wdat["Zs"], wdat["mu"], alphas)))])
            mu_d = wdat["mu"][ev_d]
            for n in cfg["exp_d"]["n_list"]:
                cnt = np.stack([wdd.responders(int(x), n, rng_d).sum(0) for x in ev_d])
                curves[n].append([cos(h((cnt + a * mu_d) / (n + a), V_d), hp) for a in alphas])
        rec_d["alpha_loro_adv"], rec_d["alpha_loro_brier"] = a_adv, a_bri
        rec_d["cos_Eg_prior_only"] = cos(h(mu_d, V_d), hp)
        for n in cfg["exp_d"]["n_list"]:
            cv = np.mean(curves[n], 0)
            ia = [alphas.index(a) for a in a_adv]
            ib = [alphas.index(a) for a in a_bri]
            rec_d["by_n"][str(n)] = {
                "cos_Eg_by_alpha": cv.tolist(), "cos_Eg_consensus": float(cv[0]),
                "alpha_best": alphas[int(np.argmax(cv))], "cos_Eg_best": float(cv.max()),
                "cos_Eg_loro_adv": float(np.mean([curves[n][d][i] for d, i in enumerate(ia)])),
                "cos_Eg_loro_brier": float(np.mean([curves[n][d][i] for d, i in enumerate(ib)]))}
        res_d.append(rec_d)
        print(f"D sigma_thread={st}: alpha adv {a_adv} brier {a_bri}; prior-only cos {rec_d['cos_Eg_prior_only']:.3f}; " +
              "; ".join(f"n={n}: cons {v['cos_Eg_consensus']:.3f} adv {v['cos_Eg_loro_adv']:.3f} "
                        f"brier {v['cos_Eg_loro_brier']:.3f} best {v['cos_Eg_best']:.3f}@{v['alpha_best']}"
                        for n, v in rec_d["by_n"].items()), flush=True)

    # ---------------------------------------------------------------- C: binary group-std range (Prop. 4c)
    ms = np.arange(1, G)
    s_bin = np.sqrt(ms * (G - ms) / (G * (G - 1)))
    res_c = {"G": G, "binary_nonzero_std": dict(zip(map(str, ms.tolist()), s_bin.tolist())),
             "binary_std_min": float(s_bin.min()), "binary_std_max": float(s_bin.max()),
             "binary_std_max_over_min": float(s_bin.max() / s_bin.min()),
             "bound_G_over_2sqrt_Gm1": float(G / (2 * np.sqrt(G - 1))),
             "fractional_one_step_std": {str(n): float((1 / n) / np.sqrt(G)) for n in (2, 8, 50, 400)}}

    elapsed = time.time() - t0
    result = {"config": cfg, "n_source": world.n_source, "eval_threads": int(B), "items": int(M),
              "reference_split_half_cosine": mc_check, "rb_identity_max_rel_err": rb_err,
              "eb_alpha": [{k: v for k, v in e.items()} for e in eb], "alpha_grid": alphas,
              "exp_a": res_a, "exp_b": res_b, "exp_c": res_c, "exp_d": res_d, "elapsed_s": elapsed}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else ""
    (OUT / f"sim_results{tag}.json").write_text(json.dumps(result, indent=1))
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    (OUT / f"manifest{tag}.json").write_text(json.dumps({
        "script": "remark/scripts/sim_consensus_theory.py", "config": str(CFG.relative_to(REPO)),
        "inputs": [world.n_source], "git": git.stdout.strip(), "elapsed_s": elapsed,
        "outputs": [f"sim_results{tag}.json", f"fig_reward_variance{tag}.pdf", f"fig_adv_norm{tag}.pdf"]}, indent=1))
    make_figures(res_a, res_b, G, tag)
    print(f"done in {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------- figures
COL = {"single": "#2a78d6", "consensus": "#eb6834", "shrink_eb": "#1baf7a", "shrink_u": "#eda100"}
LAB = {"single": "single responder", "consensus": "consensus", "shrink_eb": "shrinkage, LORO-adv $\\alpha$",
       "shrink_u": "shrinkage, LORO-Brier $\\alpha$"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def make_figures(res_a, res_b, G, tag) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib unavailable ({e}); figures skipped")
        return
    plt.rcParams.update({"font.size": 8, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
    rtypes = ("single", "consensus", "shrink_eb", "shrink_u")
    series = [Line2D([], [], color=COL[r], lw=2, marker="o", ms=4, label=LAB[r]) for r in rtypes]
    ns = [r["n"] for r in res_a]
    sm = [r["summary"] for r in res_a]

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axs[0]
    for e in rtypes:
        ax.errorbar(ns, [s[f"cos_pop_{e}"] for s in sm], yerr=[s[f"cos_pop_{e}_sd"] for s in sm], color=COL[e],
                    lw=2, marker="o", ms=4, capsize=2)
    ax.plot(ns, [s["cos_Eg_pop_consensus"] for s in sm], color=COL["consensus"], lw=1, ls="--")
    ax.plot(ns, [s["cos_Eg_pop_shrink_eb"] for s in sm], color=COL["shrink_eb"], lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("responders per thread $n$")
    ax.set_ylabel("cosine to $h_{pop}$")
    ax.set_title("(a) per-step update (dashed: $E[\\hat g]$)", fontsize=8)
    ax = axs[1]
    ax.plot(ns, [s["var_ratio_single_over_consensus"] for s in sm], color=COL["single"], lw=2, marker="o", ms=4)
    ax.plot(ns, [1 / (1 - s["frac_var_from_responder_sampling"]) for s in sm], color=INK2, lw=1, ls="--")
    ax.text(ns[-1], 1.1, "dashed: $1/(1-f)$,\n$f$ = responder share\nof tr Var($\\hat g_S$)", color=INK2, fontsize=7,
            ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("responders per thread $n$")
    ax.set_ylabel("tr Var($\\hat g_S$) / tr Var($\\hat g_C$)")
    ax.set_title("(b) variance removed by consensus", fontsize=8)
    ax = axs[2]
    ax.plot(ns, [s["mse_ratio_eb"] for s in sm], color=COL["shrink_eb"], lw=2, marker="o", ms=4)
    ax.plot(ns, [s["mse_ratio_u"] for s in sm], color=COL["shrink_u"], lw=2, marker="o", ms=4)
    ax.plot(ns, [s["mse_ratio_best_alpha"] for s in sm], color=INK2, lw=1, ls="--")
    ax.axhline(1.0, color=COL["consensus"], lw=1, ls=":")
    ax.text(ns[0], 1.02, "plug-in $c/n$", color=INK2, fontsize=7, va="bottom")
    ax.text(ns[-1], 0.05, "dashed: MSE-best $\\alpha$", color=INK2, fontsize=7, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("responders per thread $n$")
    ax.set_ylabel("MSE($\\tilde p$) / MSE($\\hat p$)")
    ax.set_title("(c) shrinkage vs plug-in weights", fontsize=8)
    for a in axs:
        _style(a)
    fig.legend(handles=series, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / f"fig_reward_variance{tag}.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axs[0]
    lv = np.linspace(0, 1, 201)
    for r in rtypes:
        ax.step(np.maximum(res_b[r]["s_nonzero_ecdf_q"], 1e-6), lv, where="post", color=COL[r], lw=1.8)
    ax.axvline(1 / np.sqrt(G), color=INK2, lw=1, ls="--")
    ax.text(1 / np.sqrt(G) * 0.8, 0.05, "binary min\n$1/\\sqrt{G}$", color=INK2, fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1.0)
    ax.set_xlabel("nonzero within-group reward std $s$")
    ax.set_ylabel("fraction of groups $\\leq s$")
    ax.set_title("(a) group std, Reddit-like $n$", fontsize=8)
    _style(ax)
    x = np.arange(len(rtypes))
    for ax, key, title in ((axs[1], "cos_pop", "(b) update cosine to $h_{pop}$"),
                           (axs[2], "adv_reliability", "(c) advantage reliability")):
        for j, m in enumerate(("group", "batch")):
            ax.bar(x + (j - 0.5) * 0.38, [res_b[r][key][m] for r in rtypes], width=0.36, color=[COL[r] for r in rtypes],
                   hatch="////" if m == "group" else None, edgecolor="white", lw=0.5, alpha=0.55 if m == "group" else 1.0)
        ax.set_xticks(x, ["single", "cons.", "shr.\n(adv)", "shr.\n(Brier)"])
        ax.set_title(title, fontsize=8)
        _style(ax)
    axs[1].set_ylabel("cosine")
    axs[2].set_ylabel("corr. across responder samples")
    axs[2].legend(handles=[Patch(facecolor=INK2, alpha=0.55, hatch="////", edgecolor="white", label="group std"),
                           Patch(facecolor=INK2, label="batch std")], frameon=False, fontsize=7, loc="upper left")
    fig.legend(handles=series, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / f"fig_adv_norm{tag}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
