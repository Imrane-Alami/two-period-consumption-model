"""
The Two-Period Consumption-Saving Model
=======================================
Theory (figures 1-7) and a test on US data, 1959-2009 (figures 8-9).

Household problem:
    max  u(C_t) + beta * u(C_{t+1})
    s.t. C_t + C_{t+1}/(1+r) = Y_t + Y_{t+1}/(1+r) = W      (intertemporal budget constraint)

Euler equation:
    u'(C_t) = beta * (1+r) * u'(C_{t+1})

Run:  python consumption.py   -> figures are saved in figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import brentq
from statsmodels.sandbox.regression.gmm import IV2SLS

from style import (BLACK, GREEN, ORANGE, BLUE, GREY, textbook_axes, project, label,
                   caption, style_series, save)


# ------------------------------------------------------------------
# 1. The model (log utility unless stated otherwise)
# ------------------------------------------------------------------
def wealth(Yt, Yt1, r):
    """Lifetime wealth (present value of income)."""
    return Yt + Yt1 / (1 + r)


def optimum(Yt, Yt1, r, beta):
    """Log utility: C_t = W / (1+beta), and C_{t+1} = beta (1+r) C_t (Euler)."""
    Ct = wealth(Yt, Yt1, r) / (1 + beta)
    return Ct, beta * (1 + r) * Ct


def utility(Ct, Ct1, beta):
    return np.log(Ct) + beta * np.log(Ct1)


def indifference(Ct, U, beta):
    """C_{t+1} on the indifference curve of level U."""
    return np.exp((U - np.log(Ct)) / beta)


def budget(Ct, Yt, Yt1, r):
    """C_{t+1} = (1+r)(Y_t - C_t) + Y_{t+1}."""
    return (1 + r) * (Yt - Ct) + Yt1


def hicks_point(U0, r, beta):
    """Bundle on the old indifference curve with the new slope -(1+r): pure substitution effect."""
    ln_ct = (U0 - beta * np.log(beta * (1 + r))) / (1 + beta)
    Ct = np.exp(ln_ct)
    return Ct, beta * (1 + r) * Ct


# Check: the closed form satisfies both the budget constraint and the Euler equation
_c0, _c1 = optimum(4, 8, 0.05, 0.95)
assert abs(_c0 + _c1 / 1.05 - wealth(4, 8, 0.05)) < 1e-12
assert abs(1 / _c0 - 0.95 * 1.05 / _c1) < 1e-12

BETA, R = 0.95, 0.05


def plot_indiff(ax, U, beta, xmax, ymax, **kw):
    c = np.linspace(0.02 * xmax, xmax, 400)
    ci = indifference(c, U, beta)
    keep = ci <= ymax * 1.02
    ax.plot(c[keep], ci[keep], **kw)
    return c[keep], ci[keep]


# ------------------------------------------------------------------
# Fig. 1 — The intertemporal budget constraint
# ------------------------------------------------------------------
def fig_budget():
    Yt, Yt1, r = 5, 5, 0.25
    xmax, ymax = wealth(Yt, Yt1, r), (1 + r) * Yt + Yt1
    c = np.linspace(0, xmax, 100)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    textbook_axes(ax, r"$C_t$", r"$C_{t+1}$", xmax, ymax)
    ax.plot(c, budget(c, Yt, Yt1, r), color=BLACK, lw=1.6)
    ax.plot(Yt, Yt1, "o", color=BLACK, ms=4)
    project(ax, Yt, Yt1, r"$Y_t$", r"$Y_{t+1}$")
    ax.text(Yt * 1.04, Yt1 * 1.04, "endowment", fontsize=10)
    ax.text(-xmax * 0.015, ymax, r"$(1+r)Y_t+Y_{t+1}$", ha="right", va="center")
    ax.text(xmax, -ymax * 0.035, r"$Y_t+\dfrac{Y_{t+1}}{1+r}$", ha="center", va="top")
    ax.text(0.38 * Yt, 0.25 * ymax, "saver\n" + r"$S_t>0$", color=BLUE, ha="center", fontsize=10)
    ax.text(1.5 * Yt, 0.25 * ymax, "borrower\n" + r"$S_t<0$", color=ORANGE, ha="center", fontsize=10)
    ax.text(0.22 * xmax, budget(0.22 * xmax, Yt, Yt1, r) * 1.03, r"slope $-(1+r)$", fontsize=10)
    caption(fig, "Fig. 1: The intertemporal budget constraint always passes through the endowment point.")
    save(fig, "01_budget_constraint")


# ------------------------------------------------------------------
# Fig. 2 — Indifference curves and the optimum
# ------------------------------------------------------------------
def fig_optimum():
    Yt, Yt1, r, beta = 4, 8, R, BETA
    W = wealth(Yt, Yt1, r)
    Ct, Ct1 = optimum(Yt, Yt1, r, beta)
    xmax, ymax = W * 1.02, (1 + r) * Yt + Yt1
    U_star = utility(Ct, Ct1, beta)
    fig, ax = plt.subplots(figsize=(6, 5))
    textbook_axes(ax, r"$C_t$", r"$C_{t+1}$", xmax, ymax)
    c = np.linspace(0, W, 100)
    ax.plot(c, budget(c, Yt, Yt1, r), color=BLACK, lw=1.6)
    for U, nm in [(U_star - 0.35, r"$U_0$"), (U_star, r"$U_1$"), (U_star + 0.3, r"$U_2$")]:
        cc, ci = plot_indiff(ax, U, beta, xmax, ymax, color=GREEN, lw=1.1)
        label(ax, cc[-1] * 1.01, ci[-1], nm, GREEN, ha="left", fontsize=10)
    ax.plot(Yt, Yt1, "o", color=BLACK, ms=3.5)
    ax.text(Yt * 0.97, Yt1 * 1.05, "endowment", fontsize=9.5, ha="right")
    ax.plot(Ct, Ct1, "o", color=ORANGE, ms=5)
    project(ax, Ct, Ct1, r"$C_t^*$", r"$C_{t+1}^*$", color=ORANGE)
    ax.text(Ct * 1.05, Ct1 * 1.04, "optimum", color=ORANGE, fontsize=10)
    ax.plot([Yt, Yt], [0, Yt1], ls=(0, (2, 2)), lw=0.7, color=GREY)
    ax.text(Yt, -ymax * 0.035, r"$Y_t$", ha="center", va="top")
    caption(fig, r"Fig. 2: The optimum is the tangency point: $\dfrac{u'(C_t)}{\beta u'(C_{t+1})}=1+r$ (Euler equation)."
                 "\nHere $C_t^*>Y_t$: the household borrows.", y=-0.03)
    save(fig, "02_optimum")


# ------------------------------------------------------------------
# Fig. 3 — Higher present or future income: consumption smoothing
# ------------------------------------------------------------------
def fig_income_shocks():
    beta, r = BETA, R
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (Y0, Y1), (Y0n, Y1n), title in [
        (axes[0], (5, 5), (7, 5), r"(a) $Y_t$ rises: saving rises"),
        (axes[1], (5, 5), (5, 7), r"(b) $Y_{t+1}$ rises: saving falls"),
    ]:
        xmax = wealth(Y0n, Y1n, r) * 1.02
        ymax = (1 + r) * Y0n + Y1n
        textbook_axes(ax, r"$C_t$", r"$C_{t+1}$", xmax, ymax)
        for (a, b), col, lw in [((Y0, Y1), GREY, 1.1), ((Y0n, Y1n), BLACK, 1.6)]:
            c = np.linspace(0, wealth(a, b, r), 100)
            ax.plot(c, budget(c, a, b, r), color=col, lw=lw)
            Ct, Ct1 = optimum(a, b, r, beta)
            plot_indiff(ax, utility(Ct, Ct1, beta), beta, xmax, ymax, color=GREEN, lw=1, alpha=0.9 if col == BLACK else 0.45)
            ax.plot(Ct, Ct1, "o", color=ORANGE if col == BLACK else GREY, ms=4.5)
            ax.plot(a, b, "s", color=col, ms=3.5)
            ax.text(wealth(a, b, r), ymax * 0.03, "before" if col == GREY else "after", color=col,
                    ha="right", fontsize=9.5)
        c_old, c1_old = optimum(Y0, Y1, r, beta)
        c_new, c1_new = optimum(Y0n, Y1n, r, beta)
        ax.annotate("", (c_new, c1_new), (c_old, c1_old), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1))
        ax.text(c_new * 1.06, c1_new * 1.05, r"both $C_t$ and $C_{t+1}$ rise", color=ORANGE, fontsize=10)
        ax.set_title(title, fontsize=11, pad=12)
    fig.tight_layout(w_pad=5)
    caption(fig, "Fig. 3: Whenever income rises, the household raises consumption in both periods: "
                 "it smooths consumption by adjusting saving.")
    save(fig, "03_income_shocks_smoothing")


# ------------------------------------------------------------------
# Fig. 4 — A higher interest rate: substitution and income effects
# ------------------------------------------------------------------
def fig_interest_rate():
    beta, r0, r1 = BETA, 0.05, 0.6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, (Yt, Yt1), who in [(axes[0], (8, 2), "saver"), (axes[1], (2, 8), "borrower")]:
        xmax = max(wealth(Yt, Yt1, r0), wealth(Yt, Yt1, r1)) * 1.02
        ymax = (1 + r1) * Yt + Yt1 if who == "saver" else (1 + r0) * Yt + Yt1 + 1
        textbook_axes(ax, r"$C_t$", r"$C_{t+1}$", xmax, ymax)
        A = optimum(Yt, Yt1, r0, beta)
        C = optimum(Yt, Yt1, r1, beta)
        U0 = utility(*A, beta)
        B = hicks_point(U0, r1, beta)
        for r_, col, lw in [(r0, GREY, 1.1), (r1, BLACK, 1.6)]:
            c = np.linspace(0, wealth(Yt, Yt1, r_), 100)
            ax.plot(c, budget(c, Yt, Yt1, r_), color=col, lw=lw)
        # hypothetical line: new slope, tangent to the old indifference curve
        W_h = B[0] + B[1] / (1 + r1)
        c = np.linspace(0, W_h, 100)
        ax.plot(c, (1 + r1) * (W_h - c), color=BLACK, ls=(0, (4, 3)), lw=0.8)
        plot_indiff(ax, U0, beta, xmax, ymax, color=GREEN, lw=1)
        plot_indiff(ax, utility(*C, beta), beta, xmax, ymax, color=GREEN, lw=1, alpha=0.6)
        left = {"B"} if who == "saver" else {"C"}
        for P, nm, col in [(A, "A", GREY), (B, "B", BLACK), (C, "C", ORANGE)]:
            ax.plot(*P, "o", color=col, ms=4.5)
            dx = -0.03 * xmax if nm in left else 0.03 * xmax
            ax.text(P[0] + dx, P[1], nm, color=col, fontsize=11, va="center",
                    ha="right" if nm in left else "left")
        ax.plot(Yt, Yt1, "s", color=BLACK, ms=3.5)
        ax.annotate("", B, A, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
        ax.annotate("", C, B, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
        sign = "+" if who == "saver" else "-"
        ax.text(0.97 * xmax, 0.93 * ymax,
                "A → B: substitution effect\n" + r"($C_t\downarrow$, $C_{t+1}\uparrow$)" +
                f"\nB → C: income effect ({sign})", ha="right", va="top", fontsize=9.5)
        ax.set_title(f"({'a' if who == 'saver' else 'b'}) A {who}", fontsize=11, pad=12)
    fig.tight_layout(w_pad=5)
    caption(fig, f"Fig. 4: The interest rate rises from {r0:.0%} to {r1:.0%} (exaggerated for visibility): the budget line pivots around the "
                 "endowment.\nThe substitution effect always lowers $C_t$; the income effect helps savers and hurts borrowers.",
            y=-0.03)
    save(fig, "04_interest_rate_effects")


# ------------------------------------------------------------------
# Fig. 5 — The Euler equation and the elasticity of intertemporal substitution
# ------------------------------------------------------------------
def fig_euler_eis(sigma_hat):
    beta = 0.99
    r = np.linspace(-0.04, 0.10, 100)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style_series(ax)
    for sigma, col in [(0.5, BLUE), (1, BLACK), (2, GREEN), (sigma_hat, ORANGE)]:
        g = 100 * (np.log(beta) + r) / sigma
        ax.plot(100 * r, g, color=col, lw=1.8 if sigma == sigma_hat else 1.2,
                ls="-" if sigma != sigma_hat else (0, (5, 2)))
        txt = rf"$\sigma={sigma:g}$" if sigma != sigma_hat else rf"$\hat\sigma={sigma_hat:.1f}$ (US data)"
        label(ax, 100 * r[-1] + 0.2, g[-1], txt, col, ha="left", fontsize=10)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set(xlabel=r"real interest rate $r$ (%)", ylabel=r"consumption growth $\Delta\ln C$ (%)", xlim=(-4, 10))
    caption(fig, r"Fig. 5: CRRA Euler equation: $\Delta \ln C_{t+1} \approx \frac{1}{\sigma}(\ln\beta + r)$. "
                 r"The higher $\sigma$, the less consumption growth reacts to $r$.")
    save(fig, "05_euler_eis")


# ------------------------------------------------------------------
# Fig. 6 — Temporary vs permanent shocks: the marginal propensity to consume
# ------------------------------------------------------------------
def fig_mpc():
    r = 0.03
    T = np.arange(2, 61)
    annuity = (r / (1 + r)) / (1 - (1 + r) ** (-T))   # with beta(1+r)=1, consumption is flat
    mpc_temp = annuity                                # +1 today raises W by 1
    mpc_perm = np.ones_like(T, dtype=float)           # +1 every period raises W by the full annuity value
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style_series(ax)
    ax.plot(T, mpc_perm, color=BLACK, lw=1.6)
    ax.plot(T, mpc_temp, color=ORANGE, lw=1.6)
    label(ax, T[-1] + 1, 1, "permanent shock", BLACK, ha="left", fontsize=10)
    label(ax, T[-1] + 1, mpc_temp[-1], "temporary shock", ORANGE, ha="left", fontsize=10)
    ax.plot(2, mpc_temp[0], "o", color=ORANGE, ms=4)
    ax.annotate(f"two-period model:\nMPC = {mpc_temp[0]:.2f}", (2, mpc_temp[0]), xytext=(9, 0.62),
                fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set(xlabel="horizon $T$ (years of remaining life)", ylabel="MPC out of current income", ylim=(0, 1.1), xlim=(0, 60))
    caption(fig, r"Fig. 6: With $\beta(1+r)=1$ and $r$ = 3%, a temporary windfall is spread over the whole horizon: "
                 "the longer the horizon,\nthe lower the MPC. A permanent rise in income is consumed one for one.", y=-0.03)
    save(fig, "06_mpc_temporary_vs_permanent")
    return mpc_temp[0]


# ------------------------------------------------------------------
# Fig. 7 — Uncertainty and precautionary saving
# ------------------------------------------------------------------
def crra_mu(c, sigma):
    return c ** (-sigma)


def optimal_saving(spread, mu, Y=1.0, r=0.05):
    """Two-state income risk: Y_{t+1} = Y +/- spread. Solve u'(C_t) = beta(1+r) E[u'(C_{t+1})]."""
    beta = 1 / (1 + r)

    def foc(s):
        c_t = Y - s
        c_lo, c_hi = Y - spread + (1 + r) * s, Y + spread + (1 + r) * s
        return mu(c_t) - beta * (1 + r) * 0.5 * (mu(c_lo) + mu(c_hi))

    lo = max(-(Y - spread) / (1 + r) + 1e-6, -0.99)
    return brentq(foc, lo, Y - 1e-6)


def fig_precautionary():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # (a) Jensen: E[u'(C)] > u'(E[C]) when u' is convex
    ax = axes[0]
    c = np.linspace(0.35, 2.2, 300)
    mu = crra_mu(c, 2)
    textbook_axes(ax, r"$C_{t+1}$", r"$u'(C_{t+1})$", 2.2, crra_mu(0.35, 2))
    ax.plot(c, mu, color=BLACK, lw=1.6)
    lo, hi = 0.6, 1.6
    mid = (lo + hi) / 2
    ax.plot([lo, hi], [crra_mu(lo, 2), crra_mu(hi, 2)], color=GREY, lw=0.9)
    for x in (lo, hi):
        ax.plot(x, crra_mu(x, 2), "o", color=BLACK, ms=3.5)
    e_mu = (crra_mu(lo, 2) + crra_mu(hi, 2)) / 2
    ax.plot(mid, e_mu, "o", color=ORANGE, ms=4.5)
    ax.plot(mid, crra_mu(mid, 2), "o", color=BLUE, ms=4.5)
    project(ax, mid, e_mu, r"$E[C_{t+1}]$", r"$E[u']$", color=ORANGE)
    project(ax, mid, crra_mu(mid, 2), None, r"$u'(E[C])$", color=BLUE)
    for x, nm in [(lo, r"$C^{L}$"), (hi, r"$C^{H}$")]:
        ax.plot([x, x], [0, crra_mu(x, 2)], ls=(0, (2, 2)), lw=0.7, color=GREY)
        ax.text(x, -crra_mu(0.35, 2) * 0.035, nm, ha="center", va="top")
    ax.text(1.75, crra_mu(1.75, 2) + 0.6, r"$u'(C)=C^{-\sigma}$", fontsize=10)
    ax.set_title(r"(a) Convex $u'$: risk raises expected marginal utility", fontsize=11, pad=12)

    # (b) optimal saving vs income risk
    ax = axes[1]
    style_series(ax)
    spreads = np.linspace(0, 0.6, 61)
    theta = 0.4
    for nm, mu_f, col in [(r"CRRA, $\sigma=4$", lambda x: crra_mu(x, 4), ORANGE),
                          (r"CRRA, $\sigma=2$", lambda x: crra_mu(x, 2), BLACK),
                          (r"log ($\sigma=1$)", lambda x: crra_mu(x, 1), BLUE),
                          (r"quadratic ($u'''=0$)", lambda x: 1 - theta * x, GREEN)]:
        sav = [100 * optimal_saving(sp, mu_f) for sp in spreads]
        ax.plot(100 * spreads, sav, color=col, lw=1.5)
        label(ax, 100 * spreads[-1] + 1, sav[-1], nm, col, ha="left", fontsize=10)
    ax.set(xlabel=r"income risk: $Y_{t+1}=1\pm$ spread (%)", ylabel=r"saving rate $S_t/Y_t$ (%)", xlim=(0, 60))
    ax.set_title("(b) Precautionary saving rises with risk", fontsize=11)
    fig.tight_layout(w_pad=5)
    caption(fig, r"Fig. 7: With prudent preferences ($u'''>0$), uncertainty about future income raises saving today. "
                 "With quadratic utility,\nonly expected income matters and there is no precautionary saving "
                 r"($\beta(1+r)=1$, $Y_t=E[Y_{t+1}]=1$).", y=-0.03)
    save(fig, "07_precautionary_saving")


# ------------------------------------------------------------------
# 2. Data: US quarterly macro data, 1959Q1-2009Q3 (shipped with statsmodels)
# ------------------------------------------------------------------
def load_data():
    d = sm.datasets.macrodata.load_pandas().data
    d["date"] = pd.PeriodIndex.from_fields(year=d.year.astype(int), quarter=d.quarter.astype(int), freq="Q")
    d["dc"] = 400 * np.log(d.realcons).diff()       # annualized consumption growth (%)
    d["dy"] = 400 * np.log(d.realdpi).diff()        # annualized disposable income growth (%)
    d["r"] = d.realint                              # ex post real T-bill rate (%)
    d["saving_rate"] = 100 * (1 - d.realcons / d.realdpi)
    d[["date", "realcons", "realdpi", "realint", "tbilrate", "infl", "unemp", "saving_rate"]].to_csv(
        "data/us_macro_quarterly.csv", index=False)
    return d


def estimate(d):
    out = {}
    # (i) Euler equation: Delta ln C_{t+1} = a + (1/sigma) r_t
    x = d.assign(dc_next=d.dc.shift(-1)).dropna(subset=["dc_next", "r"])
    m = sm.OLS(x.dc_next, sm.add_constant(x.r)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    out["eis"] = (m.params["r"], m.bse["r"])
    out["euler_model"] = m
    out["euler_data"] = x
    # (ii) Hall (1978): under the permanent income hypothesis, lagged variables cannot predict Delta ln C
    h = d.assign(dc_lag=d.dc.shift(1), dy_lag=d.dy.shift(1)).dropna(subset=["dc", "dc_lag", "dy_lag"])
    mh = sm.OLS(h.dc, sm.add_constant(h[["dc_lag", "dy_lag"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    out["hall"] = {k: (mh.params[k], mh.bse[k]) for k in ["dc_lag", "dy_lag"]}
    out["hall_p"] = float(mh.f_test("dc_lag = 0, dy_lag = 0").pvalue)
    # (iii) Campbell-Mankiw (1989): share lambda of "rule-of-thumb" consumers, IV with lags 2-4
    z = pd.concat({f"{v}_l{k}": d[v].shift(k) for v in ["dc", "dy", "r"] for k in (2, 3, 4)}, axis=1)
    df = pd.concat([d[["dc", "dy", "r"]], z], axis=1).dropna()
    iv = IV2SLS(df.dc, sm.add_constant(df[["dy", "r"]]), sm.add_constant(df[z.columns])).fit()
    first = sm.OLS(df.dy, sm.add_constant(df[z.columns])).fit()
    out["lambda"] = (iv.params["dy"], iv.bse["dy"])
    out["first_stage_F"] = first.fvalue
    return out


# ------------------------------------------------------------------
# Fig. 8 — Testing the theory on US data
# ------------------------------------------------------------------
def fig_estimates(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    style_series(ax)
    x, m = res["euler_data"], res["euler_model"]
    ax.scatter(x.r, x.dc_next, s=9, color=GREY, alpha=0.7, lw=0)
    rr = np.linspace(x.r.min(), x.r.max(), 50)
    ax.plot(rr, m.params["const"] + m.params["r"] * rr, color=ORANGE, lw=1.8)
    eis, se = res["eis"]
    ax.text(0.97, 0.95, rf"slope $=1/\sigma$ = {eis:.2f} (s.e. {se:.2f})" + "\n" + rf"$\Rightarrow \sigma \approx$ {1 / eis:.1f}",
            transform=ax.transAxes, ha="right", va="top", color=ORANGE, fontsize=10)
    ax.set(xlabel=r"real interest rate $r_t$ (%, annualized)", ylabel=r"consumption growth $\Delta\ln C_{t+1}$ (%, ann.)")
    ax.set_title("(a) The Euler equation: a small EIS", fontsize=11)

    ax = axes[1]
    style_series(ax)
    rows = [("Hall: lagged $\\Delta\\ln C_{t-1}$", *res["hall"]["dc_lag"]),
            ("Hall: lagged $\\Delta\\ln Y_{t-1}$", *res["hall"]["dy_lag"]),
            (r"Campbell-Mankiw: $\lambda$", *res["lambda"])]
    for i, (nm, b, s) in enumerate(rows[::-1]):
        col = ORANGE if "lambda" in nm else BLACK
        ax.errorbar(b, i, xerr=1.96 * s, fmt="o", color=col, ms=5, capsize=3, lw=1.2)
        ax.text(b, i + 0.22, f"{b:.2f}", ha="center", fontsize=10, color=col)
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (3, 3)))
    ax.text(0.01, -0.62, "PIH prediction: 0", color=GREY, fontsize=9.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_xlabel("estimate and 95% confidence interval")
    ax.set_title("(b) Consumption reacts to predictable income", fontsize=11)
    fig.tight_layout(w_pad=4)
    caption(fig, "Fig. 8: US quarterly data, 1959-2009. Consumption growth barely reacts to the interest rate, and it is "
                 "predictable from past income,\nagainst the pure permanent income hypothesis. The Campbell-Mankiw estimate "
                 f"suggests about half of 'rule-of-thumb' consumers,\nbut its instruments are weak (first-stage F = "
                 f"{res['first_stage_F']:.1f}): it should be read with caution.", y=-0.03)
    save(fig, "08_us_euler_hall_campbell_mankiw")


# ------------------------------------------------------------------
# Fig. 9 — A natural experiment: the 2008 tax rebates
# ------------------------------------------------------------------
def fig_rebate_2008(d):
    w = d[(d.year >= 2007)].copy()
    base = w[(w.year == 2008) & (w.quarter == 1)].iloc[0]
    lab = [f"{int(y)}Q{int(q)}" for y, q in zip(w.year, w.quarter)]
    xpos = np.arange(len(w))
    q2 = lab.index("2008Q2")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    style_series(ax)
    ax.axvspan(q2 - 0.5, q2 + 0.5, color=ORANGE, alpha=0.12, lw=0)
    ax.plot(xpos, 100 * w.realdpi / base.realdpi, color=BLACK, marker="o", ms=3.5)
    ax.plot(xpos, 100 * w.realcons / base.realcons, color=BLUE, marker="o", ms=3.5)
    label(ax, xpos[-1] + 0.3, 100 * w.realdpi.iloc[-1] / base.realdpi, "disposable income", BLACK, ha="left", fontsize=10)
    label(ax, xpos[-1] + 0.3, 100 * w.realcons.iloc[-1] / base.realcons, "consumption", BLUE, ha="left", fontsize=10)
    ax.text(q2, 102.6, "rebate checks\n(2008Q2)", ha="center", color=ORANGE, fontsize=9.5)
    ax.set_xticks(xpos, lab, rotation=45)
    ax.set(ylabel="index, 2008Q1 = 100", ylim=(97.5, 103.3))
    ax.set_title("(a) Income jumps, consumption does not", fontsize=11)

    ax = axes[1]
    style_series(ax)
    ax.axvspan(q2 - 0.5, q2 + 0.5, color=ORANGE, alpha=0.12, lw=0)
    ax.plot(xpos, w.saving_rate, color=BLACK, marker="o", ms=3.5)
    ax.set_xticks(xpos, lab, rotation=45)
    ax.set(ylabel="saving rate (% of disposable income)")
    ax.set_title("(b) The saving rate jumps", fontsize=11)
    fig.tight_layout(w_pad=4)

    d_inc = w.realdpi.iloc[q2] - w.realdpi.iloc[q2 - 1]
    d_con = w.realcons.iloc[q2] - w.realcons.iloc[q2 - 1]
    caption(fig, f"Fig. 9: In 2008Q2, real disposable income rose by \\${d_inc:.0f}bn (annual rate) while consumption rose "
                 f"by \\${d_con:.1f}bn: in the aggregate,\nthe immediate MPC out of this temporary windfall looks close to "
                 f"{d_con / d_inc:.0%}, in line with Fig. 6. But the quarter is confounded by the oil shock and the "
                 "onset of the recession;\nhousehold-level studies exploiting the random timing of the checks find "
                 "much larger spending responses.", y=-0.05)
    save(fig, "09_us_2008_rebate")
    return d_inc, d_con


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures…")
    data = load_data()
    res = estimate(data)
    fig_budget()
    fig_optimum()
    fig_income_shocks()
    fig_interest_rate()
    fig_euler_eis(1 / res["eis"][0])
    mpc2 = fig_mpc()
    fig_precautionary()
    fig_estimates(res)
    d_inc, d_con = fig_rebate_2008(data)

    print("\nKey results (US, 1959Q1-2009Q3):")
    print(f"  EIS (1/sigma)            : {res['eis'][0]:.2f} (s.e. {res['eis'][1]:.2f})  -> sigma ~ {1 / res['eis'][0]:.1f}")
    print(f"  Hall test, joint p-value : {res['hall_p']:.1e}  (PIH rejected)")
    print(f"  Campbell-Mankiw lambda   : {res['lambda'][0]:.2f} (s.e. {res['lambda'][1]:.2f}), first-stage F = {res['first_stage_F']:.1f}")
    print(f"  2008Q2 rebate            : income +{d_inc:.0f}bn, consumption +{d_con:.1f}bn -> MPC ~ {d_con / d_inc:.1%}")
    print(f"  Two-period MPC (theory)  : {mpc2:.2f}")
