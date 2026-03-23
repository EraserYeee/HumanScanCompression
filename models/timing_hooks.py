"""轻量耗时打点：由 Stage2Pipeline 设置子模块的 _profile_do_log 后使用。"""
import time
import torch


def profile_is_rank0():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def prof_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prof_start():
    prof_sync()
    return time.perf_counter()


def prof_split(do_log, t0, label, group="pipe"):
    if not do_log:
        return time.perf_counter()
    prof_sync()
    dt_ms = (time.perf_counter() - t0) * 1000
    if profile_is_rank0():
        print(f"[timing/{group}] {label}: {dt_ms:.2f}ms")
    return time.perf_counter()
