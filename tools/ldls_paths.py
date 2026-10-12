"""Where the driver binary and the model are, so the Python tools run unchanged on any machine.

Lookup order:
  1. the environment variables LDLS_DRIVER / LDLS_MODEL, if set
  2. the repo-local layout:  build/ldls_driver[.exe]  and  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  3. C:\\ldls\\build and C:\\ldls\\models (the original author's machine)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
EXE = ".exe" if os.name == "nt" else ""


def _find(env, candidates, what):
    if os.environ.get(env):
        path = os.environ[env]
        if not os.path.isfile(path):
            sys.exit(f"{env} is set to '{path}', but that file does not exist")
        return path
    for path in candidates:
        if os.path.isfile(path):
            return path
    tried = "\n  ".join(candidates)
    sys.exit(f"{what} not found. Looked in:\n  {tried}\nBuild/download it (see README.md) or set {env} to its path.")


def driver():
    return _find("LDLS_DRIVER", [
        os.path.join(ROOT, "build", "ldls_driver" + EXE),
        os.path.join(ROOT, "build", "Release", "ldls_driver" + EXE),  # Visual Studio generators
        r"C:\ldls\build\ldls_driver.exe",
    ], "The driver (ldls_driver)")


def model():
    return _find("LDLS_MODEL", [
        os.path.join(ROOT, "models", MODEL_FILE),
        os.path.join(r"C:\ldls\models", MODEL_FILE),
    ], f"The model ({MODEL_FILE})")
