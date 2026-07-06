from __future__ import annotations

import time


def log(*args):
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
    print(stamp, *args, flush=True)


def itstime(step, every_n_steps, total_steps, last=True, first=True, drop_close_to_last=0.25):
    close_to_last = False
    if drop_close_to_last and every_n_steps:
        close_to_last = abs(step - total_steps) < drop_close_to_last * every_n_steps
    is_step = every_n_steps and (step % every_n_steps == 0) and not close_to_last
    is_last = every_n_steps and step == total_steps
    is_first = every_n_steps and step == 1
    return bool(is_step or (last and is_last) or (first and is_first))
