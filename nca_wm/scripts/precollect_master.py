#!/usr/bin/env python3
"""Pre-collect (state, action, next_state) transitions for the deduped master
corpus, sharded for a SLURM array.

Mirrors `precollect_dedup_pool.py` but targets the flat gist master dir
(`puzzlescript-gists`) instead of the curated repo corpora: it registers the
master dir as an extra games dir so the C++ backend resolves gist-hash stems by
name, reads the kept candidates from `dedup_master.json`, and only collects the
parseable ones (`parse_status == "ok"`).

DEFAULT ALGORITHM IS BFS, not A*. For world-model training we want broad,
representative coverage of the dynamics rather than the goal corridor, and BFS
avoids A*'s priority-queue (O(log frontier) per push/pop) + heuristic cost under
the same max_iters/timeout ceiling. Pass --search_algo astar to compare.

Caches land at rollout_data/<stem>/level_<i>/<algo>_transitions_*.npz, the exact
format the bucketed loader in train.py reads. Already-cached (game, level) pairs
are skipped instantly, so a preempted array job resumes for free.

SHARDING: pass --shard K --num-shards M to process only candidates whose index
% M == K. One SLURM array task per shard; all write to the shared rollout_data.

Usage (local smoke):
    .venv/bin/python3 -m nca_wm.scripts.precollect_master \
        --master-dir ../puzzlescript-gists --limit 3 --workers 1

Usage (one shard, as run under SLURM array):
    .venv/bin/python3 -m nca_wm.scripts.precollect_master \
        --master-dir /scratch/se2161/puzzlescript-gists \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 128 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Randomness sources in PuzzleScript rules: a `random` / `random[dir]` rule
# prefix, a `random`/`randomdir` RHS modifier, or a `random` object spawn from a
# legend group. All make the transition non-deterministic, which confounds a
# deterministic world model — so we exclude these games for the multi-game WM.
_RANDOM_RE = re.compile(r"\brandom(dir)?\b", re.IGNORECASE)
_RULES_HDR = re.compile(r"^RULES\s*$", re.IGNORECASE | re.MULTILINE)
_WIN_HDR = re.compile(r"^WINCONDITIONS\s*$", re.IGNORECASE | re.MULTILINE)


def _has_randomness(master_dir: str, stem: str) -> bool:
    """True if the game's RULES section uses any randomness construct."""
    p = Path(master_dir) / (stem + ".txt")
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    # Restrict the scan to the RULES section so object/legend names that merely
    # contain "random" don't trip it. Comments may yield a rare false positive,
    # which is the safe direction (we want randomness OUT).
    m = _RULES_HDR.search(txt)
    if not m:
        return bool(_RANDOM_RE.search(txt))
    end = _WIN_HDR.search(txt, m.end())
    rules = txt[m.end():end.start() if end else len(txt)]
    return bool(_RANDOM_RE.search(rules))


def _build_pool(args) -> list[dict]:
    """Kept, parseable candidates from dedup_master.json, in a stable order."""
    report = json.loads(Path(args.manifest).read_text())
    master_dir = str(Path(args.manifest).resolve().parent)
    out = []
    for c in report["candidates"]:
        if c.get("parse_status") != "ok":
            continue  # only parseable games can be rolled out by the engine
        out.append(c)
    # Grid-size filtering happens per-level in the worker (--max_cells), since
    # the manifest has no level-area field for the gist corpus.
    # Stable order: simplest first (fewest objects, then levels, then name) so a
    # partial run still yields the cheap, high-yield games. Objects drive the
    # channel count / per-transition cost more than level count does.
    out.sort(key=lambda c: (c.get("n_objs") or 0, c.get("n_levels") or 0, c["file"]))
    if args.num_shards > 1:
        out = [c for i, c in enumerate(out) if i % args.num_shards == args.shard]
    if args.limit:
        out = out[:args.limit]
    # Randomness filter (after sharding, so each shard only stats its own files).
    if args.skip_random:
        n_before = len(out)
        out = [c for c in out
               if not _has_randomness(master_dir, c["file"][:-4])]
        n_rand = n_before - len(out)
        if n_rand:
            print(f"  randomness filter: dropped {n_rand} non-deterministic "
                  f"games ({len(out)} remain)", flush=True)
    return out


def _game_already_cached(stem: str, n_levels: int, search_algo: str) -> bool:
    # max_iters and the cap are now chosen adaptively per level, so they appear
    # in the cache filename and can't be predicted here. Treat a level as cached
    # if ANY transition cache for this algo exists (we never write more than one
    # per (game, level)). Levels skipped for size simply never produce a file;
    # such games re-enter the worker and are re-skipped instantly.
    import glob
    cache_dir_root = REPO_ROOT / "rollout_data" / stem
    if not cache_dir_root.is_dir():
        return False
    for li in range(max(1, n_levels)):
        pattern = (f"{cache_dir_root}/level_{li}/"
                   f"{search_algo}_transitions_v*_cap*.npz")
        if not glob.glob(pattern):
            return False
    return True


def _rss_gb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1e9
    except Exception:
        return -1.0


def _collect_one(meta: dict, args, repo_root: str, master_dir: str,
                 on_level_start=None) -> tuple[str, str]:
    """Worker: register master dir, compile the gist stem, collect every level.

    on_level_start(li) is invoked before each level so an external watchdog can
    reset a per-level deadline (used by the isolated scheduler).
    """
    sys.path.insert(0, repo_root)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    # Hold-but-don't-use the GPU: hide it from JAX/CUDA so no pinned host memory
    # is allocated per worker (suspected torch-only OOM source; CPU-only work).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from nca_wm.train import collect_unique_transitions
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.utils import init_ps_lark_parser
    from puzzlescript_jax.preprocessing import add_extra_games_dir

    # Resolve gist-hash stems (e.g. "9d573d3cfa79cfa1f132") by name.
    add_extra_games_dir(master_dir)

    if not hasattr(_collect_one, "_ps_parser"):
        _collect_one._ps_parser = init_ps_lark_parser()
    ps_parser = _collect_one._ps_parser

    stem = meta["file"][:-4] if meta["file"].endswith(".txt") else meta["file"]
    backend = CppPuzzleScriptBackend()
    try:
        json_str = backend.compile_and_serialize(ps_parser, stem)
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    except Exception as e:
        return stem, f"FAIL_compile: {type(e).__name__}: {str(e)[:120]}"
    n_levels = env0.num_levels
    if n_levels < 1:
        return stem, "FAIL_no_levels"
    if args.log_rss:
        print(f"    [start] {stem} n_levels={n_levels} "
              f"n_objs_canon={int(env0.observation_shape[0])}", flush=True)

    # PEAK-RAM control. collect_unique_transitions has TWO large transient
    # tensors per level: (1) the full visited set as int32 before the write cap,
    # ~ (5*max_iters) * cells * ceil(n_objs/32) * 4B * 2; and (2) the dense
    # multihot it expands the *kept* transitions to, ~ cap * n_objs * cells * 2B.
    # Both scale with cells AND n_objs, so a flat cap OOMs on high-object games
    # at 16 workers (the n_per_rule pipeline only survived because it ran
    # workers=1). We instead bound each level's estimated peak to --mem_budget_mb
    # by scaling max_iters and max_transitions down per level: small games keep
    # full settings, only large/high-object games are throttled.
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    N = max(1, int(env0.observation_shape[0]))   # canonical channels (multihot out_C)
    # RAW object count drives the packed state width (stride = ceil(raw/32)),
    # which is what the C++ result arrays AND the multihot intermediate `arr`
    # actually allocate — often >> canonical N when many objects dedup to few
    # channels. Sizing the RAM budget by canonical N (instead of raw) was the
    # cause of repeated OOMs. Fetch raw count from the engine.
    if args.max_objs is not None and N > args.max_objs:
        return stem, f"SKIP_too_many_objs (n_objs={N} > {args.max_objs})"
    budget = args.mem_budget_mb * 1_000_000       # per-worker peak target
    # The C++ result is first materialized as a Python list of ints (~28B each)
    # then copied to int32 (4B); both coexist during np.asarray, so the transient
    # costs ~32B per packed word. The multihot pass also builds a uint32 `arr` of
    # width cells*stride_raw. Both phases bounded to `budget` below; tiny floors
    # so the budget — not the floor — binds even for high-stride / large grids.
    B_INT = 40  # ~ int32 array (4B) + transient Python-list int (28B) + ptr/overhead
    default_cap = (None if args.max_transitions_per_game is None
                   else max(1, args.max_transitions_per_game // max(1, n_levels)))
    t0 = time.time()
    n_collected = 0
    n_skipped_big = 0
    biggest = 0
    for li in range(n_levels):
        if on_level_start is not None:
            on_level_start(li)
        try:
            probe = Engine()
            probe.load_from_json(json_str)
            probe.load_level(li)
            raw_objs = max(1, int(probe.get_object_count()))
            S_raw = max(1, (raw_objs + 31) // 32)     # packed-int stride (RAW)
            # ACTUAL packed state width (= cells * stride) the collector stores
            # per transition. Using this instead of get_width()*get_height()
            # is immune to ragged levels (padded to max width -> huge state but
            # mis-reported dims) and viewport/zoomscreen mismatches — the cause
            # of a >110G transient on an under-measured ragged level.
            state_words = max(1, int(len(probe.get_objects())))
            cells = max(1, state_words // S_raw)
            biggest = max(biggest, cells)
            del probe
        except Exception:
            cells = args.max_cells or 1024  # if unsizable, assume cap-sized
            raw_objs = max(1, N); S_raw = max(1, (raw_objs + 31) // 32)
            state_words = cells * S_raw
        if args.max_cells is not None and cells > args.max_cells:
            n_skipped_big += 1
            continue
        # (1) transient: (5*iters) packed states of width=state_words, ~40B/word
        #     (Python list-of-ints during np.asarray + the int32 copy).
        iters_cap = int(budget / (5 * state_words * B_INT))
        iters = max(100, min(args.n_search_steps, iters_cap))
        # (2) multihot phase per kept transition, x2 (state+next): a uint32 `arr`
        #     of state_words*4B plus a uint8 out of N*cells.
        mh_cap = int(budget / (2 * (state_words * 4 + N * cells)))
        cap = max(200, mh_cap if default_cap is None else min(default_cap, mh_cap))
        try:
            data = collect_unique_transitions(
                json_str, stem, level_i=li,
                max_iters=iters,
                timeout_ms=args.search_timeout_ms,
                search_algo=args.search_algo,
                max_transitions=cap,
            )
            n_collected += len(data["actions"])
            del data
        except Exception as e:
            return stem, f"FAIL_l{li}: {type(e).__name__}: {str(e)[:120]}"
        if args.log_rss:
            import resource as _r
            peak = _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1_048_576
            print(f"    [rss] {stem} level {li}/{n_levels} cells={cells} "
                  f"raw_objs={raw_objs} N={N} iters={iters} cap={cap} "
                  f"RSS={_rss_gb():.2f}G peak={peak:.2f}G", flush=True)
    dt = time.time() - t0
    # Per-game peak RSS. With max_tasks_per_child=1 each worker process handles
    # exactly one game, so RUSAGE_SELF peak == this game's footprint (incl. the
    # ~0.36G import baseline). Reported so we can size workers from real peaks.
    import resource
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_048_576
    if n_skipped_big == n_levels:
        return stem, f"SKIP_all_levels_too_big (max_cells={biggest}) ({dt:.1f}s)"
    tag = "" if not n_skipped_big else f" skip_big={n_skipped_big}"
    return stem, (f"OK n_levels={n_levels - n_skipped_big}/{n_levels} "
                  f"n_trans={n_collected}{tag} peak={peak_gb:.1f}G ({dt:.1f}s)")


def _stem_of(meta) -> str:
    f = meta["file"]
    return f[:-4] if f.endswith(".txt") else f


def _isolated_child(meta, args, repo_root, master_dir, q):
    """Child process: run one game, posting a per-level heartbeat to `q` (at each
    level boundary, when the GIL is free) so the PARENT can enforce limits even
    when this process is blocked in a GIL-holding C++/numpy section. The parent
    SIGKILLs us on a stale heartbeat (per-level timeout) or RSS breach — no
    in-child thread is used (it couldn't run while the GIL is held)."""
    stem = _stem_of(meta)

    def _on_level(li):
        try:
            q.put(("HB", stem, li))
        except Exception:
            pass

    try:
        s, status = _collect_one(meta, args, repo_root, master_dir,
                                  on_level_start=_on_level)
    except MemoryError:
        s, status = stem, "FAIL_oom (MemoryError)"
    except Exception as e:
        s, status = stem, f"FAIL_worker: {type(e).__name__}: {str(e)[:100]}"
    q.put(("R", s, status))


def _proc_rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1e9
    except Exception:
        return 0.0


def _run_isolated(todo, args, repo_root, master_dir):
    """Bounded scheduler with PARENT-enforced per-level timeout + RSS cap.

    Each game runs in a fork()ed child. The parent SIGKILLs (terminate) a child
    whose latest heartbeat is older than --per_level_timeout (search hang, even
    GIL-stuck in C++) or whose RSS exceeds --mem_cap_gb (any mis-sized cap /
    ragged-level blowup). A killed child is recorded FAIL_* and the next game
    launched — the node never OOMs and the shard never hangs."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    todo_iter = iter(todo)
    running = {}            # proc -> {meta, hb_t, li}
    results = {}            # stem -> status
    n_total = len(todo)
    n_done = n_ok = n_fail = 0

    def _launch():
        meta = next(todo_iter, None)
        if meta is None:
            return False
        p = ctx.Process(target=_isolated_child,
                        args=(meta, args, repo_root, master_dir, q))
        p.start()
        running[p] = {"meta": meta, "hb_t": time.time(), "li": -1}
        return True

    def _by_stem(stem):
        for pp, info in running.items():
            if _stem_of(info["meta"]) == stem:
                return pp
        return None

    for _ in range(max(1, args.workers)):
        if not _launch():
            break

    while running:
        # drain heartbeats + results
        while not q.empty():
            msg = q.get()
            if msg[0] == "HB":
                pp = _by_stem(msg[1])
                if pp is not None:
                    running[pp]["hb_t"] = time.time()
                    running[pp]["li"] = msg[2]
            else:  # ("R", stem, status)
                results[msg[1]] = msg[2]
        now = time.time()
        for p in list(running):
            info = running[p]
            stem = _stem_of(info["meta"])
            killed = None
            if p.is_alive():
                rss = _proc_rss_gb(p.pid)
                if rss > args.mem_cap_gb:
                    killed = f"FAIL_oom (RSS {rss:.1f}G > {args.mem_cap_gb:.0f}G)"
                elif now - info["hb_t"] > args.per_level_timeout:
                    killed = (f"FAIL_timeout (level {info['li']} "
                              f">{args.per_level_timeout:.0f}s)")
                if killed:
                    p.terminate()
                    p.join()
                else:
                    continue
            else:
                p.join()
            running.pop(p, None)
            st = killed or results.pop(stem, None)
            if st is None:
                st = f"FAIL_killed (exitcode={p.exitcode})"
            n_done += 1
            ok = st.startswith("OK")
            n_ok += ok
            n_fail += not ok
            print(f"[{n_done}/{n_total}] {stem}: {st}", flush=True)
            _launch()
        time.sleep(0.3)
    return n_done, n_ok, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-dir", default=str(REPO_ROOT.parent / "puzzlescript-gists"))
    ap.add_argument("--manifest", default=None,
                    help="dedup_master.json (default <master>/dedup_master.json)")
    ap.add_argument("--search_algo", default="bfs", choices=("bfs", "astar"))
    # The collector materializes the FULL visited transition set in RAM before
    # the write-time cap, so peak RAM ~= workers * (5*max_iters) * 2 * cells *
    # ceil(n_objs/32) * 4B. 60k iters (=>300k transitions/level) keeps a 32x32
    # board safe at 16 workers / 110G, while preserving broad coverage (most
    # games exhaust their reachable states far below this).
    ap.add_argument("--n_search_steps", type=int, default=60000)
    ap.add_argument("--search_timeout_ms", type=int, default=60000)
    ap.add_argument("--max_transitions_per_game", type=int, default=200000)
    ap.add_argument("--max_cells", type=int, default=1024,
                    help="Skip any level with W*H grid cells above this "
                         "(1024 ~= 32x32). None disables.")
    ap.add_argument("--max_objs", type=int, default=384,
                    help="Skip games with more than this many object channels "
                         "(untrainable beyond the model's MAX_CHANNELS_V2=384, "
                         "and n_objs drives peak RAM). None disables.")
    ap.add_argument("--mem_budget_mb", type=int, default=3000,
                    help="Per-worker peak-RAM target. max_iters and "
                         "max_transitions are scaled down per level so each "
                         "transient tensor fits this. Tune with --workers so "
                         "workers * budget stays under the SLURM --mem request.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_cached", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip_random", action=argparse.BooleanOptionalAction, default=True,
                    help="Exclude games with random/randomdir rules "
                         "(non-deterministic; confound a deterministic WM).")
    ap.add_argument("--log_rss", action=argparse.BooleanOptionalAction, default=False,
                    help="Print per-level RSS (diagnostic).")
    ap.add_argument("--mem_cap_gb", type=float, default=10.0,
                    help="Hard per-game RSS ceiling. A watchdog thread kills the "
                         "game's process if its RSS exceeds this (handles any "
                         "game the adaptive cap mis-sizes). Keep workers*cap < node mem.")
    ap.add_argument("--per_level_timeout", type=float, default=180.0,
                    help="Hard per-LEVEL wall-clock ceiling. The watchdog kills "
                         "the process if one level exceeds this (catches games "
                         "whose search hangs, e.g. malformed/ragged levels, even "
                         "when stuck in C++ where SIGALRM can't interrupt).")
    args = ap.parse_args()

    master_dir = str(Path(args.master_dir).resolve())
    if args.manifest is None:
        args.manifest = str(Path(master_dir) / "dedup_master.json")

    games = _build_pool(args)
    print(f"master: {master_dir}")
    print(f"shard {args.shard}/{args.num_shards}: {len(games)} parseable candidates"
          f"  algo={args.search_algo}  workers={args.workers}", flush=True)

    todo = []
    for g in games:
        stem = g["file"][:-4]
        n_levels = int(g.get("n_levels") or 1)
        if args.skip_cached and _game_already_cached(
                stem, n_levels, args.search_algo):
            continue
        todo.append(g)
    n_skipped = len(games) - len(todo)
    print(f"  cached+skipped: {n_skipped}   to collect: {len(todo)}", flush=True)

    repo_root = str(REPO_ROOT)
    t0 = time.time()
    n_done = n_ok = n_fail = 0
    if args.workers <= 1:
        for g in todo:
            stem, status = _collect_one(g, args, repo_root, master_dir)
            n_done += 1
            n_ok += status.startswith("OK"); n_fail += not status.startswith("OK")
            print(f"[{n_done}/{len(todo)}] {stem}: {status}", flush=True)
    else:
        # HARD-ISOLATED scheduler: each game runs in its own fork()ed process
        # with an in-process watchdog thread that os._exit()s if the game's RSS
        # exceeds --mem_cap_gb OR any single level exceeds --per_level_timeout.
        # os._exit fires even when the main thread is stuck in C++ (where SIGALRM
        # can't), so a pathological game (ragged/huge level, runaway search, any
        # mis-sized cap) dies ALONE and is recorded FAIL_* — the node never OOMs
        # and the shard never hangs. At most `workers` games run concurrently;
        # process death frees all of a game's RAM. No perfect memory model needed.
        n_done, n_ok, n_fail = _run_isolated(todo, args, repo_root, master_dir)

    print(f"\nDONE shard {args.shard}/{args.num_shards}  total={n_done} ok={n_ok} "
          f"fail={n_fail} skipped={n_skipped}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
