"""Process-tree helpers (stdlib only): suspend, resume, kill, is-alive.

On Windows a venv's python.exe is a launcher that runs the real interpreter as a child, so acting on the pid alone would
leave the child (and its GPU memory) running: every operation here covers the pid and all its descendants.
"""
from __future__ import annotations

import os, signal, subprocess, sys

WINDOWS = sys.platform == "win32"

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    PROCESS_SUSPEND_RESUME, PROCESS_QUERY_LIMITED_INFORMATION = 0x0800, 0x1000
    TH32CS_SNAPPROCESS, STILL_ACTIVE = 0x2, 259

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE

    def _parents() -> dict[int, int]:
        snap = _k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        out, e = {}, PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = _k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out[e.th32ProcessID] = e.th32ParentProcessID
            ok = _k32.Process32NextW(snap, ctypes.byref(e))
        _k32.CloseHandle(snap)
        return out

    def _each(pids, fn):
        for pid in pids:
            h = _k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
            if h:
                fn(wintypes.HANDLE(h)); _k32.CloseHandle(h)


def tree(pid: int) -> list[int]:
    """pid and all its descendants, parents first."""
    if WINDOWS:
        parents = _parents()
        out, frontier = [pid], [pid]
        while frontier:
            kids = [c for c, p in parents.items() if p in frontier and c not in out]
            out += kids; frontier = kids
        return out
    try:
        kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        kids = []
    return [pid] + [d for k in kids for d in tree(int(k))]


def suspend(pid: int):
    if WINDOWS: _each(tree(pid), _ntdll.NtSuspendProcess)
    else:
        for p in tree(pid): os.kill(p, signal.SIGSTOP)


def resume(pid: int):
    if WINDOWS: _each(tree(pid), _ntdll.NtResumeProcess)
    else:
        for p in reversed(tree(pid)): os.kill(p, signal.SIGCONT)


def kill(pid: int):
    if WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        for p in reversed(tree(pid)):
            try: os.kill(p, signal.SIGCONT); os.kill(p, signal.SIGKILL)
            except ProcessLookupError: pass


def alive(pid: int | None) -> bool:
    if not pid: return False
    if WINDOWS:
        h = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h: return False
        code = wintypes.DWORD()
        ok = _k32.GetExitCodeProcess(wintypes.HANDLE(h), ctypes.byref(code))
        _k32.CloseHandle(h)
        return bool(ok) and code.value == STILL_ACTIVE
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True
