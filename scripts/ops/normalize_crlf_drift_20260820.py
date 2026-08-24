#!/usr/bin/env python3
"""
One-off fix for the recurring deploy_watcher Telegram spam: every 10-minute
cycle's `git stash pop` was failing with "local changes would be overwritten
by merge" against the same ~50 tracked .py files. Root cause is *not*
someone's in-progress work -- every one of those files has CRLF line
endings on disk (the repo's own .gitattributes declares `*.py text eol=lf`,
so this is drift, not policy), and stripping \\r\\n -> \\n makes every single
one byte-identical to its git HEAD blob (verified: 50/50 files, zero real
content differences). Something wrote them with CRLF outside of any git
operation, bypassing the eol=lf normalization that only kicks in on git's
own checkout/add/commit.

This script re-verifies that invariant per file (abort loudly rather than
touch a file whose CRLF-stripped content does NOT match HEAD -- that would
mean it has real changes mixed in, which is out of scope here and needs a
human) and only then rewrites it with LF endings, restoring a clean
`git status` so future deploy_watcher cycles have nothing to stash.

Does not touch untracked files (a different, much larger set that may be
real in-progress research work) and does not touch the git stash list --
run scripts/ops/report_stale_deploy_watcher_stashes_20260820.py separately
to see whether those are safe to drop.
"""
import subprocess
import sys

REPO_ROOT = "/home/llewyn/crypto-scalping"


def main():
    files = subprocess.run(
        ["git", "diff", "--name-only"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.splitlines()
    if not files:
        print("no modified tracked files -- nothing to do")
        return

    print(f"{len(files)} modified tracked files found")
    fixed = []
    skipped_clean = []
    aborted = []
    for f in files:
        try:
            with open(f"{REPO_ROOT}/{f}", "rb") as fh:
                working = fh.read()
        except Exception as e:
            aborted.append((f, f"read_error:{e}"))
            continue
        head = subprocess.run(
            ["git", "show", f"HEAD:{f}"], capture_output=True, cwd=REPO_ROOT
        ).stdout
        normalized = working.replace(b"\r\n", b"\n")
        head_normalized = head.replace(b"\r\n", b"\n")
        if normalized != head_normalized:
            aborted.append((f, "content_differs_from_HEAD_even_after_crlf_normalize -- NOT touched, has real changes"))
            continue
        if normalized == working:
            skipped_clean.append(f)
            continue
        with open(f"{REPO_ROOT}/{f}", "wb") as fh:
            fh.write(normalized)
        fixed.append(f)

    print(f"normalized to LF: {len(fixed)}")
    print(f"already LF (no-op): {len(skipped_clean)}")
    print(f"ABORTED (real content differences, left untouched): {len(aborted)}")
    for f, reason in aborted:
        print(f"  {f}: {reason}")

    remaining = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout
    remaining_tracked = [
        line for line in remaining.splitlines() if line and not line.startswith("??")
    ]
    print()
    print(f"git status after fix: {len(remaining_tracked)} tracked file(s) still modified")
    for line in remaining_tracked:
        print(" ", line)
    if aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
