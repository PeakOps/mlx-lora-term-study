"""B deneyi korumasi (A'nin pipeline_paper/common.py kalibi): ses kapali mi, canli istemci var mi, swap, isi kapisi.
Her yogun adimdan once guard() cagrilir. Ses ayarlarina DOKUNMAZ, yalniz okur."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SWAP_LIMIT_MB = 2048
LOG = Path(__file__).resolve().parents[1] / "logs" / "guard.jsonl"


def muted() -> bool:
    r = subprocess.run(["osascript", "-e", "output muted of (get volume settings)"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "true"


def live_clients() -> list[str]:
    r = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:ESTABLISHED"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split()
        if len(f) < 9 or "->" not in f[8]:
            continue
        loc, rem = f[8].split("->")
        if any(loc.endswith(p) for p in (":8770", ":8767", ":8765")) and not rem.startswith("127.0.0.1"):
            out.append(f[8])
    return out


def swap_used_mb() -> float:
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    # "total = 2048.00M  used = 584.25M  free = ..."
    return float(s.split("used =")[1].split()[0].rstrip("M").replace(",", "."))  # tr yerel ayar: ondalik virgul


def macmon() -> dict:
    try:
        out = subprocess.run(["macmon", "pipe", "-s", "1", "-i", "800"],
                             capture_output=True, text=True, timeout=8).stdout.strip().splitlines()[0]
        d = json.loads(out)
        t = d.get("temp", {})
        return {"cpu_c": round(t.get("cpu_temp_avg", -1), 1), "gpu_c": round(t.get("gpu_temp_avg", -1), 1),
                "all_w": round(d.get("all_power", 0), 1), "gpu_w": round(d.get("gpu_power", 0), 1)}
    except Exception:  # noqa: BLE001
        return {}


class Stop(Exception):
    pass


def guard(tag: str = "", hard_exit: bool = True) -> dict:
    reason = None
    if not muted():
        reason = "hoparlor sessiz degil"
    elif live_clients():
        reason = f"canli istemci bagli {live_clients()}"
    elif swap_used_mb() > SWAP_LIMIT_MB:
        reason = f"swap {swap_used_mb():.0f} MB > {SWAP_LIMIT_MB}"
    if reason:
        print(f"[guard] {tag} {reason} -> DUR", flush=True)
        if hard_exit:
            sys.exit(3)
        raise Stop(reason)
    t0 = time.time()
    if os.environ.get("THERMAL_GATE"):  # optional thermal gate script, called as: <script> 80 70
        subprocess.run(["zsh", os.environ["THERMAL_GATE"], "80", "70"])
    m = macmon()
    m.update(tag=tag, cool_wait_s=round(time.time() - t0, 1), swap_mb=swap_used_mb(), ts=time.time())
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(m) + "\n")
    print(f"[guard] {tag} {m}", flush=True)
    return m


def a_running() -> bool:
    """A deneyi surecleri (pipeline_paper e*.py) calisiyor mu? ps ile yalniz okunur."""
    r = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    return any("pipeline_paper" in ln or "/e3_latency.py" in ln or "e4_memory.py" in ln or "e1_" in ln
               for ln in r.splitlines() if "grep" not in ln)
