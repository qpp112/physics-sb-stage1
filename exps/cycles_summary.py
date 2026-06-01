#!/usr/bin/env python3
"""Leaderboard from cycles.log.

Primary (fair H2): rank by trained PSNR vs GT -- comparable across references.
Secondary: gain vs own static (legacy; favors weak brownian prior).
"""
import re
from pathlib import Path

log = Path(__file__).resolve().parents[1] / "cycles.log"
if not log.exists():
    print("no cycles.log"); raise SystemExit(1)


def _f(s):
    if s is None or s == "nan":
        return float("nan")
    return float(s)


PAT_NEW = re.compile(
    r"RESULT \[(?P<verdict>WIN|lose|DOMINATE)\] variant=(?P<variant>\S+) ref=(?P<ref>\S+) "
    r"size=(?P<size>\d+) steps=(?P<steps>\d+) n_train=(?P<n_train>\d+) "
    r"wd=(?P<wd>[0-9.]+) drop=(?P<drop>[0-9.]+) noise=(?P<noise>[0-9.]+) "
    r"base_ch=(?P<base_ch>\d+) loss_fn=(?P<loss_fn>\S+) lr_sched=(?P<lr_sched>\S+) .*"
    r"trained=(?P<trained>[0-9.]+) static=(?P<static>[0-9.]+) "
    r"gain=(?P<gain>[+-]?[0-9.]+)dB hi_gain=(?P<hi_gain>[+-]?[0-9.]+|nan)dB "
    r"hi_trained=(?P<hi_trained>[0-9.]+|nan) resid_closed=(?P<resid_closed>[0-9.]+)"
)
PAT_OLD = re.compile(
    r"RESULT \[(?P<verdict>WIN|lose|DOMINATE)\] variant=(?P<variant>\S+) ref=(?P<ref>\S+) "
    r"steps=(?P<steps>\d+) n_train=(?P<n_train>\d+) wd=(?P<wd>[0-9.]+) drop=(?P<drop>[0-9.]+) "
    r"noise=(?P<noise>[0-9.]+).*"
    r"trained=(?P<trained>[0-9.]+) static=(?P<static>[0-9.]+) gain=(?P<gain>[+-]?[0-9.]+)dB"
)

text = log.read_text()
rows = []
for line in text.splitlines():
    m = PAT_NEW.search(line) or PAT_OLD.search(line)
    if not m:
        continue
    d = m.groupdict()
    rows.append(dict(
        verdict=d["verdict"], variant=d["variant"], ref=d["ref"],
        size=int(d.get("size") or 64),
        steps=int(d["steps"]), n_train=int(d["n_train"]),
        wd=float(d["wd"]), drop=float(d["drop"]), noise=float(d["noise"]),
        base_ch=int(d.get("base_ch") or 32),
        loss_fn=d.get("loss_fn") or "mse",
        lr_sched=d.get("lr_sched") or "none",
        trained=float(d["trained"]), static=float(d["static"]),
        gain=float(d["gain"]),
        hi_gain=_f(d.get("hi_gain")),
        hi_trained=_f(d.get("hi_trained")),
        resid_closed=_f(d.get("resid_closed")),
    ))

PAIRED = re.compile(
    r"PAIRED_RESULT size=(?P<size>\d+) steps=(?P<steps>\d+) n_train=(?P<n_train>\d+) "
    r"wd=(?P<wd>[0-9.]+) drop=(?P<drop>[0-9.]+) loss=(?P<loss>\S+) lr=(?P<lr>\S+) \| "
    r"phys_tr=(?P<phys>[0-9.]+) brwn_tr=(?P<brwn>[0-9.]+) mean_delta=(?P<delta>[+-]?[0-9.]+) "
    r"hi_delta=(?P<hi_delta>[+-]?[0-9.]+|nan) phys_wins=(?P<wins>\d+)/(?P<n>\d+) verdict=(?P<verdict>\S+)"
)
FULL = re.compile(
    r"FULL_PAIRED_RESULT size=(?P<size>\d+) seed=(?P<seed>\d+) steps=(?P<steps>\d+) "
    r"n_test=(?P<n_test>\d+) \| mean_delta=(?P<delta>[+-]?[0-9.]+) "
    r"hi_delta=(?P<hi_delta>[+-]?[0-9.]+|nan) "
    r"phys_psnr=(?P<phys>[0-9.]+) brwn_psnr=(?P<brwn>[0-9.]+) "
    r"phys_ssim=(?P<phys_ssim>[0-9.]+) brwn_ssim=(?P<brwn_ssim>[0-9.]+) "
    r"phys_lpips=(?P<phys_lp>[0-9.]+) brwn_lpips=(?P<brwn_lp>[0-9.]+) "
    r"phys_epe=(?P<phys_epe>[0-9.]+) brwn_epe=(?P<brwn_epe>[0-9.]+) "
    r"wins=(?P<wins>\d+)/(?P<n>\d+) prereg=(?P<prereg>\S+)"
)
paired = []
full_paired = []
for line in text.splitlines():
    m = PAIRED.search(line)
    if m:
        paired.append({k: m.group(k) for k in m.groupdict()})
    m2 = FULL.search(line)
    if m2:
        full_paired.append({k: m2.group(k) for k in m2.groupdict()})

if not rows and not paired and not full_paired:
    print("no RESULT lines yet"); raise SystemExit(0)


def _print_table(title, key, fmt):
    print(f"=== {title} ===\n")
    print(f"  {'rank':>4}  {key:>10}  {'ref':8}  {'px':>3}  {'loss':5}  {'n_tr':>5}  "
          f"{'gain':>7}  {'hi_tr':>6}  {'rc':>5}  steps")
    for i, r in enumerate(sorted(rows, key=lambda x: x[key], reverse=True)[:12], 1):
        hi = f"{r['hi_trained']:.2f}" if r["hi_trained"] == r["hi_trained"] else " n/a"
        rc = f"{r['resid_closed']:.2f}" if r["resid_closed"] == r["resid_closed"] else " n/a"
        print(f"  {i:4d}  {fmt(r[key]):>10}  {r['ref']:8}  {r['size']:3d}  {r['loss_fn']:5}  "
              f"{r['n_train']:5}  {r['gain']:+.3f}  {hi:>6}  {rc:>5}  {r['steps']}")


if rows:
    _print_table("Trained PSNR vs GT (unpaired runs; check config match)", "trained",
                 lambda v: f"{v:.3f}dB")
    print()
    _print_table("Legacy gain vs own static (NOT fair across references)", "gain",
                 lambda v: f"{v:+.3f}dB")

phys = max((r for r in rows if r["ref"] == "physics"), key=lambda r: r["trained"], default=None)
brwn = max((r for r in rows if r["ref"] == "brownian"), key=lambda r: r["trained"], default=None)
if paired:
    print("\n=== FAIR PAIRED H2 (same data, init, batch order) ===\n")
    print(f"  {'px':>3}  {'steps':>5}  {'mean_d':>8}  {'hi_d':>8}  {'wins':>10}  verdict")
    for p in sorted(paired, key=lambda x: float(x["delta"]), reverse=True):
        hi = p["hi_delta"] if p["hi_delta"] != "nan" else "   n/a"
        print(f"  {p['size']:>3}  {p['steps']:>5}  {float(p['delta']):+8.3f}  "
              f"{hi:>8}  {p['wins']}/{p['n']:>6}  {p['verdict']}")
    print("\n  mean_delta = per-sample PSNR(physics) - PSNR(brownian) on identical test clips")

if full_paired:
    print("\n=== FULL FAIR 256px (1000 clips, all metrics, prereg gate) ===\n")
    print(f"  {'seed':>4}  {'mean_d':>8}  {'hi_d':>8}  {'PSNR_p':>7}  {'PSNR_b':>7}  "
          f"{'SSIM':>12}  {'LPIPS':>12}  {'EPE':>12}  prereg")
    for p in full_paired:
        hi = p["hi_delta"] if p["hi_delta"] != "nan" else "   n/a"
        ssim_d = float(p["phys_ssim"]) - float(p["brwn_ssim"])
        lp_d = float(p["phys_lp"]) - float(p["brwn_lp"])
        epe_d = float(p["phys_epe"]) - float(p["brwn_epe"])
        print(f"  {p['seed']:>4}  {float(p['delta']):+8.3f}  {hi:>8}  "
              f"{float(p['phys']):7.3f}  {float(p['brwn']):7.3f}  "
              f"{ssim_d:+.4f}      {lp_d:+.4f}      {epe_d:+.3f}      {p['prereg']}")

if rows and phys and brwn:
    print("\n=== Unpaired best-run comparison (legacy; may mismatch config) ===")
    print(f"  physics:  {phys['trained']:.3f} dB  brownian: {brwn['trained']:.3f} dB  "
          f"delta {phys['trained'] - brwn['trained']:+.3f} dB")
    print(f"  (gain would say {phys['gain'] - brwn['gain']:+.3f} dB -- not fair across refs)")
