#!/usr/bin/env python3
"""Validate the syntax of all configuration files"""
import subprocess, sys, os

def check(name, cmd, cwd=None):
    print(f"[CHECK] {name} ...")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if r.returncode != 0:
            print("  ✗ FAIL")
            print("  STDOUT:", r.stdout[-500:])
            print("  STDERR:", r.stderr[-500:])
            return False
        print("  ✓ OK")
        return True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        return False

ok = True
os.chdir("/data/workspace")

# 1. docker-compose syntax (if docker is available)
if subprocess.run("which docker", shell=True, capture_output=True).returncode == 0:
    ok &= check("docker-compose config", "docker compose config --profiles")
else:
    print("[SKIP] docker not available, skipping compose check")

# 2. pyproject.toml (use tomli for Python <3.11, tomllib for >=3.11)
try:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    with open("pyproject.toml", "rb") as f:
        tomllib.load(f)
    print("[CHECK] pyproject.toml ...\n  ✓ OK")
except ImportError:
    # fallback: basic check using pytoml / plain string parsing
    with open("pyproject.toml") as f:
        content = f.read()
    assert "[" in content and "=" in content, "invalid toml"
    print("[CHECK] pyproject.toml ...\n  ✓ OK (fallback parse)")
except Exception as e:
    print(f"[CHECK] pyproject.toml ...\n  ✗ FAIL: {e}")
    ok = False

# 3. bash script syntax
for sh in ["setup.sh", "scripts/download.sh", "gpu/entrypoint.sh"]:
    ok &= check(f"bash -n {sh}", f"bash -n {sh}")

# 4. GPU dependency file exists and is non-empty
gpu_reqs = "gpu/requirements-gpu.txt"
if os.path.isfile(gpu_reqs) and os.path.getsize(gpu_reqs) > 0:
    print(f"[CHECK] {gpu_reqs} ...\n  ✓ OK")
else:
    print(f"[CHECK] {gpu_reqs} ...\n  ✗ FAIL: file missing or empty")
    ok = False

if ok:
    print("\n===== ALL CHECKS PASSED =====")
    sys.exit(0)
else:
    print("\n===== SOME CHECKS FAILED =====")
    sys.exit(1)
