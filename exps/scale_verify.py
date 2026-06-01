#!/usr/bin/env python3
"""
Resolution scalability gate: 64 -> 128 -> (probe) 256 before full-plan commit.

Checks:
  1. H2 signal at 128px fair-paired (mean_delta >= min_delta, win rate)
  2. Cross-resolution trend 64->128 (delta and gain must not collapse)
  3. Optional cheap 256px fair probe (short train) before 12k x 1000 full run

Exit 0 = SCALABLE (proceed to full 256px plan)
Exit 1 = NOT SCALABLE (see strategies in JSON + stdout)

Usage:
  python exps/scale_verify.py --log cycles.log
  python exps/scale_verify.py --probe256 --probe_steps 4000
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRED = re.compile(
    r"PAIRED_RESULT size=(?P<size>\d+) steps=(?P<steps>\d+) n_train=(?P<n_train>\d+) "
    r"wd=(?P<wd>[0-9.]+) drop=(?P<drop>[0-9.]+) loss=(?P<loss>\S+) lr=(?P<lr>\S+) \| "
    r"phys_tr=(?P<phys>[0-9.]+) brwn_tr=(?P<brwn>[0-9.]+) mean_delta=(?P<delta>[+-]?[0-9.]+) "
    r"hi_delta=(?P<hi_delta>[+-]?[0-9.]+|nan) phys_wins=(?P<wins>\d+)/(?P<n>\d+) verdict=(?P<verdict>\S+)"
)
RESULT_PHYS = re.compile(
    r"RESULT \[\S+\] variant=refine_resid ref=physics size=(?P<size>\d+) steps=(?P<steps>\d+) "
    r"n_train=(?P<n_train>\d+) wd=(?P<wd>[0-9.]+) drop=(?P<drop>[0-9.]+) .*"
    r"trained=(?P<trained>[0-9.]+) static=(?P<static>[0-9.]+) "
    r"gain=(?P<gain>[+-]?[0-9.]+)dB"
)


def _parse_log(path: Path):
    paired, phys_rows = [], []
    for line in path.read_text().splitlines():
        m = PAIRED.search(line)
        if m:
            paired.append({k: m.group(k) for k in m.groupdict()})
        m2 = RESULT_PHYS.search(line)
        if m2:
            phys_rows.append({k: m2.group(k) for k in m2.groupdict()})
    return paired, phys_rows


def _best_paired(paired, size: int, prefer_loss="charb"):
    cands = [p for p in paired if int(p["size"]) == size]
    if not cands:
        return None
    charb = [p for p in cands if p["loss"] == prefer_loss]
    pool = charb if charb else cands
    return max(pool, key=lambda p: float(p["delta"]))


def _best_phys_gain(rows, size: int):
    cands = [r for r in rows if int(r["size"]) == size]
    if not cands:
        return None
    return max(cands, key=lambda r: float(r["gain"]))


def run_probe256(steps: int, base_ch: int, batch: int, eval_batch: int,
                 n_test: int, log_path: Path):
    """Cheap fair paired @ 256px; appends PAIRED_RESULT to cycles.log."""
    cmd = [
        sys.executable, str(ROOT / "exps/fair_pair.py"),
        "--size", "256", "--steps", str(steps), "--n_train", "3000",
        "--n_test", str(n_test), "--weight_decay", "0.10", "--dropout", "0.2",
        "--loss", "charb", "--lr_schedule", "cosine", "--input_noise", "0.0",
        "--batch", str(batch), "--eval_batch", str(eval_batch),
        "--base_ch", str(base_ch),
    ]
    print(f"[scale] 256px probe: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    with log_path.open("a") as f:
        f.write(f"\n########## SCALE_PROBE256 base_ch={base_ch} steps={steps} "
                f"batch={batch} eval_batch={eval_batch} ##########\n")
        for line in out.splitlines():
            if re.search(r"PAIRED_RESULT|Error|Traceback|fair\]", line):
                f.write(line + "\n")
    print(out[-4000:] if len(out) > 4000 else out)
    if proc.returncode != 0:
        raise RuntimeError(f"probe256 failed rc={proc.returncode}")
    m = PAIRED.search(out)
    if not m:
        raise RuntimeError("probe256 finished but no PAIRED_RESULT in output")
    return {k: m.group(k) for k in m.groupdict()}


def build_strategies(failures: list[str], p128, p64, gain64, gain128) -> list[dict]:
    strategies = []
    if any("delta_128" in f or "wins_128" in f for f in failures):
        strategies.append({
            "id": "fix_h2_at_128",
            "action": "More 128px fair cycles: steps=12000, n_train=6000, or mse+charb ensemble",
            "cost": "~2h GPU",
        })
    if any("trend" in f for f in failures):
        strategies.append({
            "id": "increase_steps_with_res",
            "action": "Scale steps ~linearly with pixels: 256px use 16000-20000 (128px uses 8000-12000)",
            "cost": "full run only if probe passes",
        })
    if any("probe256" in f or "gain_128" in f for f in failures):
        strategies.append({
            "id": "widen_model_256",
            "action": "At 256px use base_ch=48 or 64 (fast_cycle default 32 is under-capacity vs base.yaml 64)",
            "cost": "~1.5x train time per step",
            "probe": "fair_pair --size 256 --base_ch 48 --steps 4000",
        })
    if any("probe256" in f for f in failures):
        strategies.append({
            "id": "progressive_finetune",
            "action": "Train 128px to convergence, finetune 256px with lr=5e-5 for 4k steps (needs --init_ckpt)",
            "cost": "medium; implement init_ckpt in fair_pair",
        })
    if p128 and p64 and float(p128["delta"]) < 0.5 * float(p64["delta"]):
        strategies.append({
            "id": "match_train_budget",
            "action": "128px fair used n_train=6000 vs 64px 3000; also try 128px with n_train=3000 for apples-to-apples",
            "cost": "1 fair run",
        })
    if gain128 and gain64 and float(gain128["gain"]) < float(gain64["gain"]) * 0.5:
        strategies.append({
            "id": "regularization_at_high_res",
            "action": "Try wd=0.05, drop=0.1 at 256px; high-res may need less dropout",
            "cost": "probe runs",
        })
    if not strategies:
        strategies.append({
            "id": "retry_probe",
            "action": "Re-run 256px probe with base_ch=48 batch=8",
            "cost": "~45min",
        })
    return strategies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="cycles.log")
    ap.add_argument("--out", default="runs/scale_verdict.json")
    ap.add_argument("--min_delta_128", type=float, default=0.30)
    ap.add_argument("--min_delta_ratio", type=float, default=0.35,
                    help="128 delta / 64 delta minimum")
    ap.add_argument("--min_gain_128", type=float, default=0.45,
                    help="physics gain vs static at 128px (dB)")
    ap.add_argument("--min_win_rate", type=float, default=0.85)
    ap.add_argument("--probe256", action="store_true")
    ap.add_argument("--probe_steps", type=int, default=4000)
    ap.add_argument("--probe_base_ch", type=int, default=32)
    ap.add_argument("--probe_batch", type=int, default=4)
    ap.add_argument("--probe_eval_batch", type=int, default=4)
    ap.add_argument("--probe_n_test", type=int, default=200)
    ap.add_argument("--probe_min_delta", type=float, default=0.22,
                    help="slightly relaxed for short 256 probe")
    ap.add_argument("--require_128_fair", action="store_true",
                    help="fail if no 128px PAIRED_RESULT yet")
    ap.add_argument("--skip_trend", action="store_true",
                    help="only check 128 + probe (no 64 comparison)")
    args = ap.parse_args()

    log_path = ROOT / args.log
    if not log_path.exists():
        print(f"no log at {log_path}"); sys.exit(1)

    paired, phys_rows = _parse_log(log_path)
    p64 = _best_paired(paired, 64)
    p128 = _best_paired(paired, 128)
    gain64 = _best_phys_gain(phys_rows, 64)
    gain128 = _best_phys_gain(phys_rows, 128)

    checks = {}
    failures = []

    if args.require_128_fair and p128 is None:
        failures.append("missing_128_fair: no PAIRED_RESULT at 128px yet")
    elif p128 is None:
        checks["128_fair"] = {"status": "pending", "note": "wait for run_cycles_fair.sh"}

    if p128:
        d128 = float(p128["delta"])
        wins = int(p128["wins"]) / int(p128["n"])
        checks["128_fair"] = {
            "mean_delta": d128, "wins": p128["wins"], "n": p128["n"],
            "phys_tr": float(p128["phys"]), "brwn_tr": float(p128["brwn"]),
            "hi_delta": p128["hi_delta"],
        }
        if d128 < args.min_delta_128:
            failures.append(f"delta_128: {d128:+.3f} < {args.min_delta_128}")
        if wins < args.min_win_rate:
            failures.append(f"wins_128: {wins:.1%} < {args.min_win_rate:.0%}")

    if not args.skip_trend and p64 and p128:
        ratio = float(p128["delta"]) / max(float(p64["delta"]), 1e-6)
        checks["64_to_128_delta_ratio"] = ratio
        if ratio < args.min_delta_ratio:
            failures.append(
                f"trend_delta: 128/64 ratio {ratio:.2f} < {args.min_delta_ratio}")

    if gain128:
        g128 = float(gain128["gain"])
        checks["128_gain"] = g128
        if g128 < args.min_gain_128:
            failures.append(f"gain_128: {g128:+.3f} < {args.min_gain_128}")

    if not args.skip_trend and gain64 and gain128:
        gr = float(gain128["gain"]) / max(float(gain64["gain"]), 1e-6)
        checks["64_to_128_gain_ratio"] = gr
        if gr < 0.5:
            failures.append(f"trend_gain: 128/64 gain ratio {gr:.2f} < 0.50")

    p256 = _best_paired(paired, 256)
    if args.probe256 and p128 is not None:
        print("[scale] running 256px fair probe...")
        try:
            p256 = run_probe256(
                args.probe_steps, args.probe_base_ch, args.probe_batch,
                args.probe_eval_batch, args.probe_n_test, log_path,
            )
            paired, _ = _parse_log(log_path)
            p256 = _best_paired(paired, 256) or p256
        except Exception as e:
            failures.append(f"probe256_error: {e}")
            p256 = None

    if p256:
        d256 = float(p256["delta"])
        checks["256_probe"] = {
            "mean_delta": d256, "steps": p256["steps"], "base_ch": args.probe_base_ch,
            "phys_tr": float(p256["phys"]), "brwn_tr": float(p256["brwn"]),
        }
        if d256 < args.probe_min_delta:
            failures.append(f"probe256: {d256:+.3f} < {args.probe_min_delta}")
    elif args.probe256 and p128 and "probe256_error" not in " ".join(failures):
        failures.append("probe256: no result after probe run")

    pending_128 = p128 is None and not any(f.startswith("missing_128") for f in failures)
    scalable = p128 is not None and len(failures) == 0
    if scalable and not args.probe256 and p256 is None:
        checks["recommendation"] = "run with --probe256 before full 256px commit"

    strategies = []
    if not scalable and not pending_128:
        strategies = build_strategies(failures, p128, p64, gain64, gain128)

    verdict = {
        "scalable": scalable,
        "pending_128": pending_128,
        "failures": failures,
        "checks": checks,
        "best_paired": {"64": p64, "128": p128, "256_probe": p256},
        "strategies": strategies,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))

    print("\n=== RESOLUTION SCALABILITY GATE ===\n")
    if p64:
        print(f"  64px fair:  delta={float(p64['delta']):+.3f}  "
              f"phys={p64['phys']}  wins={p64['wins']}/{p64['n']}")
    if p128:
        print(f"  128px fair: delta={float(p128['delta']):+.3f}  "
              f"phys={p128['phys']}  wins={p128['wins']}/{p128['n']}")
    if p256:
        print(f"  256px probe: delta={float(p256['delta']):+.3f}  "
              f"phys={p256['phys']}  wins={p256['wins']}/{p256['n']}")
    if gain128:
        print(f"  128px physics gain vs static: {float(gain128['gain']):+.3f} dB")
    if pending_128:
        print("\n  VERDICT: PENDING — waiting for 128px PAIRED_RESULT in cycles.log")
    elif scalable:
        print("\n  VERDICT: SCALABLE — proceed to full 256px")
    else:
        print("\n  VERDICT: NOT SCALABLE")
    if failures:
        print("  failures:")
        for f in failures:
            print(f"    - {f}")
    if strategies:
        print("\n  strategies (ranked):")
        for i, s in enumerate(strategies, 1):
            print(f"    {i}. [{s['id']}] {s['action']}")
    print(f"\n  saved {out_path}")
    if pending_128:
        tag, code = "PENDING", 2
    elif scalable:
        tag, code = "PASS", 0
    else:
        tag, code = "FAIL", 1
    print(f"SCALE_VERDICT scalable={tag} failures={len(failures)} "
          f"delta128={float(p128['delta']) if p128 else 'n/a'}")
    sys.exit(code)


if __name__ == "__main__":
    main()
