"""Run `python -m kev.train` on Windows.

kev.train imports two Unix-only modules: `resource` (only to log the peak memory at the end) and, through kev.suite,
`fcntl` (a file lock around suite caches). Neither affects training, so this wrapper installs stand-ins and then runs
kev.train as __main__, leaving the deployed Kev checkout untouched. Run it with Kev's interpreter:

    <kev>/.venv/Scripts/python.exe kev_train_win.py --data ... (the kev.train arguments)
"""
import runpy, sys, types

if "resource" not in sys.modules:
    try:
        import resource  # noqa: F401 - present on Unix
    except ImportError:
        resource = types.ModuleType("resource")
        resource.RUSAGE_SELF = 0
        resource.getrusage = lambda who: types.SimpleNamespace(ru_maxrss=0)   # peak RSS is reported as 0
        sys.modules["resource"] = resource

if "fcntl" not in sys.modules:
    try:
        import fcntl  # noqa: F401
    except ImportError:
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN, fcntl.LOCK_SH = 2, 4, 8, 1
        fcntl.flock = lambda fd, op: None   # one training process at a time here: no lock needed
        sys.modules["fcntl"] = fcntl

sys.argv[0] = "kev.train"
runpy.run_module("kev.train", run_name="__main__", alter_sys=True)
