#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, json, subprocess
from datetime import datetime, UTC
from pathlib import Path

# === Repo configuration ===
REPO_DIR = Path("/Users/aqibsiddiqui/Desktop/Aletheia")
TARGET_FILE = REPO_DIR / "python" / "alethia" / "allocator.py"
PARENT_OWNER = "samansiddiqui55"
PARENT_REPO  = "Aletheia"
FORK_OWNER   = "Aqib121201"
BASE_BRANCH  = "main"
# ==========================

def sh(cmd: str, cwd=None, check=True) -> str:
    p = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd}\n{p.stderr}")
    return p.stdout.strip()

def has_open_pr_from_me() -> bool:
    try:
        out = sh(
            f'gh pr list --repo {PARENT_OWNER}/{PARENT_REPO} '
            f'--state open --json number,author,headRefName,headRepositoryOwner',
            cwd=REPO_DIR
        )
        items = json.loads(out or "[]")
        for pr in items:
            author = (pr.get("author") or {}).get("login", "")
            head_owner = (pr.get("headRepositoryOwner") or {}).get("login", "")
            if author == FORK_OWNER or head_owner == FORK_OWNER:
                return True
    except Exception:
        txt = sh(
            f'gh pr list --repo {PARENT_OWNER}/{PARENT_REPO} --state open',
            cwd=REPO_DIR, check=False
        )
        if FORK_OWNER.lower() in (txt or "").lower():
            return True
    return False

def ensure_upstream():
    rem = sh("git remote -v", cwd=REPO_DIR)
    if "upstream" not in rem:
        sh(f"git remote add upstream https://github.com/{PARENT_OWNER}/{PARENT_REPO}.git", cwd=REPO_DIR)
    sh("git fetch upstream", cwd=REPO_DIR)

def ff_main_with_upstream():
    sh(f"git checkout {BASE_BRANCH}", cwd=REPO_DIR)
    try:
        sh(f"git fetch upstream {BASE_BRANCH}", cwd=REPO_DIR)
        sh(f"git merge --ff-only upstream/{BASE_BRANCH}", cwd=REPO_DIR)
    except Exception:
        pass

def new_branch_name(prefix="meaningful"):
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"

def commit_and_open_pr(branch: str, title: str, body: str):
    sh(f'git add "{TARGET_FILE}"', cwd=REPO_DIR)
    sh(f'git commit -m "{title}" -m "{body}"', cwd=REPO_DIR)
    sh(f"git push -u origin {branch}", cwd=REPO_DIR)
    sh(
        f'gh pr create '
        f'--title "{title}" '
        f'--body "{body}" '
        f'--base {BASE_BRANCH} '
        f'--head {FORK_OWNER}:{branch} '
        f'--repo {PARENT_OWNER}/{PARENT_REPO}',
        cwd=REPO_DIR
    )
    print(f"✅ PR opened: {title}")

# -------------------- transformations --------------------
def refactor_staticmethods(src: str):
    changed = False
    for name in ["_matrix_to_allocation_dict", "_compute_agent_utilities"]:
        if re.search(rf"@staticmethod\s*\ndef\s*{name}\s*\(", src):
            continue
        m = re.search(rf"\n\s*def\s+{name}\s*\(", src)
        if m:
            src = src[:m.start()] + "\n    @staticmethod" + src[m.start():]
            changed = True
    if not changed:
        return None
    title = "Refactor: mark pure helpers as @staticmethod"
    body = ("Clarify lack of instance state by annotating pure helper methods with "
            "@staticmethod. No behavior change; improves readability and static analysis.")
    return src, title, body

def improve_validate_errors(src: str):
    pat = r"def\s+_validate_input_data\(self,.*?\):\s*\"\"\".*?\"\"\"(.*?)\n\s*def\s"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m: return None
    block = m.group(0)
    if "ndim=" in block and "shape=" in block: return None
    repl = {
        'raise ValueError("Utility matrix must be 2-dimensional")':
            'raise ValueError(f"Utility matrix must be 2-dimensional; '
            'got ndim={utilities.ndim}, shape={getattr(utilities, \'shape\', None)}")',
        'raise ValueError("Number of agent IDs must match utility matrix rows")':
            'raise ValueError(f"Number of agent IDs must match utility matrix rows; '
            'len(agent_ids)={len(agent_ids)}, rows={utilities.shape[0]}")',
        'raise ValueError("Number of resource IDs must match utility matrix columns")':
            'raise ValueError(f"Number of resource IDs must match utility matrix columns; '
            'len(resource_ids)={len(resource_ids)}, cols={utilities.shape[1]}")',
        'raise ValueError("Utility matrix contains non-finite values")':
            'raise ValueError("Utility matrix contains non-finite values (NaN/Inf)")',
        'raise ValueError("Utility matrix cannot be empty")':
            'raise ValueError(f"Utility matrix cannot be empty; shape={utilities.shape}")',
    }
    new_block, changed = block, False
    for old, new in repl.items():
        if old in new_block:
            new_block = new_block.replace(old, new); changed = True
    if not changed: return None
    title = "Improve input validation errors with shapes/details"
    body = ("Enhance `_validate_input_data` exceptions to include ndim/shape and counts. "
            "Faster debugging; no behavior change.")
    return src.replace(block, new_block), title, body

def add_debug_logging(src: str):
    if "self.logger.debug(\"_validate_input_data" in src: return None
    pat = r"(def\s+_validate_input_data\(self,.*?\):\n)(\s*)\"\"\"(.*?)\"\"\"\n"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m: return None
    indent = m.group(2) or "    "
    inj = (
        f"{m.group(1)}{indent}\"\"\"{m.group(3)}\"\"\"\n"
        f"{indent}if self.verbose:\n"
        f"{indent}    self.logger.debug(\"_validate_input_data: utilities.shape=%s\", getattr(utilities, 'shape', None))\n"
        f"{indent}    self.logger.debug(\"_validate_input_data: agents=%d resources=%d\", len(agent_ids), len(resource_ids))\n"
    )
    title = "Add structured debug logs in _validate_input_data"
    body = ("Lightweight debug statements (behind `self.verbose`) to expose shapes and counts "
            "during validation. No behavior change; aids troubleshooting.")
    return src.replace(m.group(0), inj), title, body

def add_raises_section(src: str):
    pat = r"(def\s+_validate_input_data\(self,.*?\):\n\s*\"\"\".*?)(\n\s*\"\"\"\n)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m: return None
    doc = m.group(1)
    if "Raises:" in doc: return None
    addition = (
        "\n\n        Raises:\n"
        "            ValueError: If the utility matrix has the wrong rank/shape,\n"
        "                contains non-finite values, or is empty.\n"
    )
    title = "Docs: document exceptions in _validate_input_data"
    body = "Add an explicit `Raises` section to improve API docs and IDE help."
    return src.replace(m.group(0), doc + addition + m.group(2)), title, body

def tidy_doc_headings(src: str):
    touched = False
    def norm(pattern, repl_text, s):
        nonlocal touched
        s2 = re.sub(pattern, repl_text, s)
        if s2 != s: touched = True
        return s2
    new_src = src
    new_src = norm(r"\bArguments:\b", "Args:", new_src)
    new_src = norm(r"\bParameters:\b", "Args:", new_src)
    new_src = norm(r"\bReturn[s]?:\b", "Returns:", new_src)
    new_src = norm(r"\bRaise[s]?:\b", "Raises:", new_src)
    if not touched: return None
    title = "Docs: normalize docstring headings"
    body = "Normalize headings (Args/Returns/Raises) for consistent Google-style docs."
    return new_src, title, body

def fallback_doc_note(src: str):
    pat = r"(class\s+BaseAllocator\(ABC\):\s*\n\s*\"\"\".*?)(\n\s*\"\"\"\n)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m: return None
    doc = m.group(1)
    if "Thread-safety note" in doc: return None
    addition = (
        "\n\n    Thread-safety note:\n"
        "        Instances are not inherently thread-safe; prefer separate instances\n"
        "        per worker when using parallel execution.\n"
    )
    title = "Docs: clarify BaseAllocator thread-safety note"
    body = "Document that allocator instances are not inherently thread-safe across threads."
    return src.replace(m.group(0), doc + addition + m.group(2)), title, body

TRANSFORMS = [
    refactor_staticmethods,
    improve_validate_errors,
    add_debug_logging,
    add_raises_section,
    tidy_doc_headings,
    fallback_doc_note,
]

def main():
    if has_open_pr_from_me():
        print("ℹ️ You already have an OPEN PR to the parent repo. Not creating another.")
        return
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Target file not found: {TARGET_FILE}")

    ensure_upstream()
    ff_main_with_upstream()

    src = TARGET_FILE.read_text(encoding="utf-8")
    result = None
    for tf in TRANSFORMS:
        out = tf(src)
        if out:
            result = out
            break

    if not result:
        print("Nothing meaningful left to change. Exiting without PR.")
        return

    new_src, title, body = result
    TARGET_FILE.write_text(new_src, encoding="utf-8")

    branch = new_branch_name("meaningful")
    sh(f"git checkout -b {branch}", cwd=REPO_DIR)
    commit_and_open_pr(branch, title, body)

if __name__ == "__main__":
    main()
