#!/usr/bin/env python3
"""서버 Windows 호스트의 블루스크린 원인 진단. 서버의 WSL2 안에서 돌린다.

WSL2 는 `/mnt/c` 로 Windows 파일시스템을 보고 `powershell.exe` 를 호출할 수 있다.
그래서 리눅스 쪽에서 BSOD 증거를 전부 읽을 수 있다:
  ① C:\\Windows\\Minidump\\*.dmp  -- 헤더를 직접 파싱해 BugCheck 코드·인자 4개
  ② 이벤트 로그 BugCheck(1001) · Kernel-Power(41/42/107) · WHEA-Logger · nvlddmkm(TDR)
  ③ CrashControl 레지스트리 -- 덤프 수집이 켜져 있는지(꺼져 있으면 다음 번에도 증거가 없다)

미니덤프는 이벤트 로그가 돌아가도 남으므로 ①과 ②를 **둘 다** 본다.
"""
from __future__ import annotations
import re, struct, subprocess, sys
from pathlib import Path

MINIDUMP = Path("/mnt/c/Windows/Minidump")
MEMDMP = Path("/mnt/c/Windows/MEMORY.DMP")

# DUMP_HEADER64 오프셋 (문서화된 구조)
OFF = {"sig": 0x00, "valid": 0x04, "machine": 0x30, "nproc": 0x34,
       "bugcheck": 0x38, "p1": 0x40, "p2": 0x48, "p3": 0x50, "p4": 0x58}

BUGCHECKS = {
    0x0000009F: ("DRIVER_POWER_STATE_FAILURE", "드라이버가 절전/복귀 전환 처리 실패 — 반복 다운의 전형"),
    0x00000116: ("VIDEO_TDR_FAILURE", "GPU 드라이버 타임아웃 복구 실패 — 장시간 GPU 부하와 직결"),
    0x00000117: ("VIDEO_TDR_TIMEOUT_DETECTED", "GPU 응답 없음"),
    0x00000124: ("WHEA_UNCORRECTABLE_ERROR", "하드웨어 — CPU·메모리·PCIe·전원부"),
    0x0000001A: ("MEMORY_MANAGEMENT", "메모리 관리자 — RAM 또는 드라이버"),
    0x00000133: ("DPC_WATCHDOG_VIOLATION", "드라이버가 DPC 를 너무 오래 점유 — 스토리지 드라이버 흔함"),
    0x0000000A: ("IRQL_NOT_LESS_OR_EQUAL", "드라이버가 잘못된 주소 접근"),
    0x000000D1: ("DRIVER_IRQL_NOT_LESS_OR_EQUAL", "드라이버 — 파일명이 같이 나오면 그게 범인"),
    0x000000EF: ("CRITICAL_PROCESS_DIED", "핵심 프로세스 종료"),
    0x0000007E: ("SYSTEM_THREAD_EXCEPTION_NOT_HANDLED", "드라이버 예외"),
    0x00000050: ("PAGE_FAULT_IN_NONPAGED_AREA", "RAM 또는 드라이버"),
    0x000000C4: ("DRIVER_VERIFIER_DETECTED_VIOLATION", "드라이버 검증기"),
    0x000000F7: ("DRIVER_OVERRAN_STACK_BUFFER", "드라이버 스택 오버런"),
    0x000000C2: ("BAD_POOL_CALLER", "드라이버 풀 오용"),
}


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"(실행 실패: {e})"


def ps(script, timeout=90):
    """WSL 에서 Windows PowerShell 호출. interop 이 꺼져 있으면 실패한다."""
    return sh(f"powershell.exe -NoProfile -NonInteractive -Command \"{script}\"", timeout)


def parse_dump(p: Path):
    try:
        b = p.read_bytes()[:0x100]
    except Exception as e:  # noqa: BLE001
        return {"file": p.name, "error": str(e)}
    if len(b) < 0x60:
        return {"file": p.name, "error": "헤더가 짧다"}
    sig = b[0:4].decode("latin1", "replace")
    valid = b[4:8].decode("latin1", "replace")
    code = struct.unpack_from("<I", b, OFF["bugcheck"])[0]
    prm = [struct.unpack_from("<Q", b, OFF[f"p{i}"])[0] for i in (1, 2, 3, 4)]
    name, hint = BUGCHECKS.get(code, ("(미등록 코드)", ""))
    return {"file": p.name, "sig": sig + valid, "code": code, "name": name,
            "hint": hint, "params": prm,
            "mtime": __import__("datetime").datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ")}


def main() -> int:
    print("=" * 78)
    print("① 미니덤프 — 블루스크린이 실제로 몇 번 있었나")
    print("=" * 78)
    if not MINIDUMP.exists():
        print(f"🔴 {MINIDUMP} 가 없다. 덤프 수집이 꺼져 있거나 BSOD 가 없었다(③ 확인).")
    else:
        dumps = sorted(MINIDUMP.glob("*.dmp"), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"미니덤프 {len(dumps)}개" + (" -- 반복 블루스크린이다." if len(dumps) > 1 else ""))
        for p in dumps[:10]:
            d = parse_dump(p)
            if "error" in d:
                print(f"  {d['file']}: 🔴{d['error']}"); continue
            print(f"\n  ▸ {d['file']}  ({d['mtime']})  [{d['sig']}]")
            print(f"    BugCheck 0x{d['code']:08X}  **{d['name']}**")
            if d["hint"]:
                print(f"      → {d['hint']}")
            print("    인자 " + " ".join(f"0x{x:016X}" for x in d["params"]))
    if MEMDMP.exists():
        d = parse_dump(MEMDMP)
        print(f"\nMEMORY.DMP 있음 ({MEMDMP.stat().st_size/1e9:.1f}GB) · "
              f"BugCheck 0x{d.get('code',0):08X} {d.get('name','')}")

    print("\n" + "=" * 78)
    print("② 이벤트 로그")
    print("=" * 78)
    out = ps("Get-WinEvent -FilterHashtable @{LogName='System';Id=1001} -MaxEvents 5 "
             "-ErrorAction SilentlyContinue | Where-Object {$_.ProviderName -like '*Bugcheck*'} | "
             "Format-List TimeCreated,Message")
    print("--- BugCheck(1001) ---\n" + (out.strip() or "(없음)"))
    out = ps("Get-WinEvent -FilterHashtable @{LogName='System';Id=41,42,107} -MaxEvents 14 "
             "-ErrorAction SilentlyContinue | Format-Table TimeCreated,Id,"
             "@{n='의미';e={switch($_.Id){41{'비정상 종료'}42{'절전 진입'}107{'절전 복귀'}}}} -Auto")
    print("\n--- Kernel-Power (41 비정상종료 / 42 절전진입 / 107 복귀) ---\n" + (out.strip() or "(없음)"))
    out = ps("Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger'} "
             "-MaxEvents 6 -ErrorAction SilentlyContinue | Format-List TimeCreated,Id,Message")
    print("\n--- WHEA-Logger (하드웨어 오류) ---\n" + (out.strip() or "(없음 — 하드웨어 오류 기록 없음)"))
    out = ps("Get-WinEvent -FilterHashtable @{LogName='System'} -MaxEvents 400 -ErrorAction SilentlyContinue | "
             "Where-Object {$_.Message -match 'nvlddmkm|TDR|display driver|시간 초과'} | "
             "Select-Object -First 6 | Format-List TimeCreated,Id,Message")
    print("\n--- GPU 드라이버 / TDR ---\n" + (out.strip() or "(없음)"))
    print("\n--- 마지막 깨운 원인 ---\n" + (sh("powercfg.exe -lastwake", 30).strip() or "(없음)"))
    print("\n--- 절전을 막고 있는 것 ---\n" + (sh("powercfg.exe /requests", 30).strip() or "(없음)"))

    print("\n" + "=" * 78)
    print("③ 덤프 수집 설정 — 꺼져 있으면 다음 번에도 증거가 없다")
    print("=" * 78)
    out = ps("Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\CrashControl' | "
             "Format-List CrashDumpEnabled,AutoReboot,DumpFile,MinidumpDir,Overwrite")
    print(out.strip() or "(읽기 실패)")
    print("  CrashDumpEnabled: 0=없음 1=전체 2=커널 3=미니 7=자동")

    print("\n" + "=" * 78)
    print("④ 전원 계획 — 절전 타이머가 살아 있나")
    print("=" * 78)
    print(sh("powercfg.exe /query SCHEME_CURRENT SUB_SLEEP 2>/dev/null | head -40", 40).strip() or "(읽기 실패)")

    print("\n" + "=" * 78)
    print("⑤ 리눅스 쪽 — 여긴 보통 아무 흔적이 없다(호스트가 죽으면 VM 은 그냥 사라진다)")
    print("=" * 78)
    print(sh("uptime; echo '---'; df -h / /mnt/c 2>/dev/null | head -5; echo '--- dmesg 마지막 ---'; "
             "dmesg 2>/dev/null | tail -12"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
