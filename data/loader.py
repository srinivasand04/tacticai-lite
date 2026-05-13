"""
data/loader.py — StatsBomb corner kick data loader (v3 — raw JSON approach)
============================================================================
Root cause of the previous failure
------------------------------------
StatsBomb's FREE open data for WC2018 does NOT have freeze frames for
corner PASS events. They only exist in StatsBomb's paid "360" product.

What IS available in the free data
------------------------------------
SHOT events (type.name == 'Shot') DO include a `shot.freeze_frame` field
in the raw events JSON for ALL free competitions. This lists every visible
player's position at the exact moment the shot was taken.

Our approach (unchanged prediction task)
-----------------------------------------
1. Fetch the raw events JSON from StatsBomb's GitHub for each match.
2. Find every Corner kick.
3. Find the first Shot taken by the SAME team within 12 seconds of that corner.
   (Most corner kicks that result in shots do so within 8-10 seconds.)
4. Use the shot's freeze_frame as the spatial snapshot.
5. Add the shooter back as a node (they are NOT in the freeze_frame by default —
   StatsBomb's freeze_frame lists all OTHER visible players).
6. Set pass_end_location = shooter's location, so graph/builder.py's
   _get_label() (nearest player to pass_end_location) correctly returns
   the shooter's node index.

Result: the task remains "predict which player is the key actor in this corner
kick sequence", just using the spatial snapshot one step later (the shot frame
rather than the kick frame). Every other module (graph, models, trainer, viz,
whatif) stays exactly the same.

Why raw JSON instead of statsbombpy API?
-----------------------------------------
statsbombpy's sb.freeze_frames() was added in v1.1.3+ and references 360 data.
The older installed version doesn't have it. By fetching the raw GitHub JSON
with requests (which statsbombpy itself depends on), we bypass version
differences entirely.
"""

import pickle
import random
import warnings
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
CACHE_FILE = DATA_DIR / "corners_cache.pkl"

# StatsBomb open data raw GitHub base URL
GITHUB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Competition: FIFA World Cup 2018
COMPETITION_ID = 43
SEASON_ID      = 3

# How many seconds after a corner to look for a shot
SHOT_WINDOW_SEC = 12


# ─── Public API ───────────────────────────────────────────────────────────────

def load_corners(
    competition_id: int = COMPETITION_ID,
    season_id:      int = SEASON_ID,
    use_cache:      bool = True,
    max_matches:    int  = None,
) -> list[dict]:
    """
    Load corner→shot sequences from StatsBomb WC2018 open data.

    Each returned dict represents a corner kick that led to a shot, with the
    shot's freeze frame as the spatial snapshot. The shooter is inserted as
    a node, and pass_end_location points to their position so builder.py's
    label logic identifies them correctly.

    Returns
    -------
    List[dict] with keys:
        match_id, event_id, location (corner origin), pass_end_location
        (shooter position), team, freeze_frame (list of player dicts)
    """
    if use_cache and CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            corners = pickle.load(f)
        if len(corners) > 0:
            print(f"[loader] {len(corners)} sequences loaded from cache.")
            return corners
        print("[loader] Cache empty — re-fetching ...")

    try:
        from statsbombpy import sb
        import requests
    except ImportError as e:
        raise ImportError("Run: pip install statsbombpy requests") from e

    print(f"[loader] Fetching match list (competition={competition_id}, season={season_id}) ...")
    matches_df = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids  = matches_df["match_id"].tolist()

    if max_matches is not None:
        match_ids = match_ids[:max_matches]

    print(f"[loader] {len(match_ids)} matches found. Fetching raw event JSON ...")
    print(f"         (corner → shot within {SHOT_WINDOW_SEC}s, shots with freeze frames)\n")

    all_sequences = []
    failed = 0
    import requests as req

    for match_id in tqdm(match_ids, desc="Loading matches"):
        try:
            seqs = _extract_from_match_raw(req, match_id)
            all_sequences.extend(seqs)
        except Exception as exc:
            tqdm.write(f"  [skip] match {match_id}: {exc}")
            failed += 1

    print(f"\n[loader] Done: {len(all_sequences)} corner→shot sequences "
          f"({failed} matches errored).")

    if len(all_sequences) == 0:
        print("\n[loader] 0 sequences found. Trying extended window (20s) ...")
        # Try again with wider window
        for match_id in tqdm(match_ids[:20], desc="Retry (20s window)"):
            try:
                seqs = _extract_from_match_raw(req, match_id, window=20)
                all_sequences.extend(seqs)
            except Exception:
                pass
        print(f"[loader] Retry result: {len(all_sequences)} sequences.")

    if len(all_sequences) > 0:
        cache_to_disk(all_sequences)

    return all_sequences


def cache_to_disk(sequences: list[dict]) -> None:
    """Save to disk for fast subsequent runs."""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(sequences, f)
    print(f"[loader] Cached {len(sequences)} sequences → {CACHE_FILE}")


# ─── Raw JSON fetching ────────────────────────────────────────────────────────

def _fetch_raw_events(requests_module, match_id: int) -> list[dict]:
    """
    Fetch the raw events JSON from StatsBomb's GitHub open data.

    URL pattern: .../data/events/{match_id}.json
    This is exactly the URL statsbombpy uses internally.
    Returns a list of event dicts (the full nested structure, not flattened).
    """
    url = f"{GITHUB_RAW}/events/{match_id}.json"
    r = requests_module.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    return r.json()


def _extract_from_match_raw(
    req,
    match_id: int,
    window: int = SHOT_WINDOW_SEC,
) -> list[dict]:
    """
    From a single match's raw events JSON:
    - Collect all Corner pass events
    - Collect all Shot events that have a freeze_frame
    - For each corner, find the first shot by the same team within `window` seconds
    - Build a sequence dict using the shot's freeze_frame

    The shooter is added back as a node because StatsBomb's shot.freeze_frame
    captures all OTHER visible players but NOT the shooter themselves.
    The shooter's location comes from the shot event's own `location` field.
    """
    events = _fetch_raw_events(req, match_id)

    corners = []
    shots_with_ff = []

    for ev in events:
        etype = ev.get("type", {}).get("name", "")

        if etype == "Pass":
            pass_data = ev.get("pass", {})
            ptype = pass_data.get("type", {}).get("name", "")
            if ptype == "Corner":
                loc = ev.get("location", [])
                end_loc = pass_data.get("end_location", [])
                if isinstance(loc, list) and len(loc) >= 2:
                    corners.append(ev)

        elif etype == "Shot":
            shot_data = ev.get("shot", {})
            ff = shot_data.get("freeze_frame", [])
            if isinstance(ff, list) and len(ff) >= 4:
                shots_with_ff.append(ev)

    if not corners or not shots_with_ff:
        return []

    results = []

    for corner in corners:
        c_period   = corner.get("period")
        c_team_id  = corner.get("team", {}).get("id")
        c_min      = corner.get("minute", 0)
        c_sec      = corner.get("second", 0)
        c_time     = c_min * 60 + c_sec
        c_loc      = corner.get("location", [])[:2]

        # Find the CLOSEST shot in time by same team within window
        best_shot = None
        best_gap  = float("inf")

        for shot in shots_with_ff:
            if shot.get("period") != c_period:
                continue
            if shot.get("team", {}).get("id") != c_team_id:
                continue
            s_min = shot.get("minute", 0)
            s_sec = shot.get("second", 0)
            s_t   = s_min * 60 + s_sec
            gap   = s_t - c_time
            if 0 < gap <= window and gap < best_gap:
                best_gap  = gap
                best_shot = shot

        if best_shot is None:
            continue  # no shot from this corner

        # ── Build the sequence dict ───────────────────────────────────────────
        ff            = best_shot.get("shot", {}).get("freeze_frame", [])
        shooter_loc   = best_shot.get("location", [])

        if len(shooter_loc) < 2:
            continue

        # The shooter's position (dict from the event, not the freeze_frame)
        shooter_pos_raw = best_shot.get("position", {})
        if not isinstance(shooter_pos_raw, dict):
            shooter_pos_raw = {"name": "Unknown"}

        # Add shooter back as a node.
        # We set actor=False so the model cannot "cheat" by looking at this flag.
        # We set teammate=True because the shooter is on the attacking team.
        shooter_node = {
            "location"  : shooter_loc[:2],
            "teammate"  : True,
            "position"  : shooter_pos_raw,
            "actor"     : False,   # hidden intentionally
            "player"    : best_shot.get("player", {"name": "Shooter"}),
        }

        # Insert shooter at a random position so its index isn't always the same.
        # graph/builder._get_label() will still identify them correctly because
        # pass_end_location == shooter_loc → nearest player distance = 0.
        insert_idx = random.randint(0, len(ff))
        ff_extended = ff[:insert_idx] + [shooter_node] + ff[insert_idx:]

        results.append({
            "match_id"          : int(match_id),
            "event_id"          : str(best_shot.get("id", "")),
            "location"          : c_loc,                 # corner kick origin
            "pass_end_location" : list(shooter_loc[:2]), # shooter position = label target
            "team"              : corner.get("team", {}).get("name", "Unknown"),
            "freeze_frame"      : ff_extended,
        })

    return results


# ─── Printing helpers ─────────────────────────────────────────────────────────

def print_sample(sequences: list[dict], n: int = 2) -> None:
    """Pretty-print N sample sequence dicts."""
    for i, s in enumerate(sequences[:n]):
        print(f"\n--- Sequence {i+1} ---")
        print(f"  match_id         : {s['match_id']}")
        print(f"  event_id         : {s['event_id']}")
        print(f"  team             : {s['team']}")
        print(f"  corner from      : {s['location']}")
        print(f"  shooter at       : {s['pass_end_location']}")
        print(f"  players in frame : {len(s['freeze_frame'])}")
        for p in s["freeze_frame"][:2]:
            name = p.get("player", {}).get("name", "?") if isinstance(p.get("player"), dict) else "?"
            pos  = p.get("position", {}).get("name", "?") if isinstance(p.get("position"), dict) else "?"
            print(f"    {name:<25s}  {pos:<20s}  teammate={p.get('teammate', '?')}")


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import requests

    print("=" * 60)
    print("TacticAI-Lite — Data Loader v3 (raw JSON, shot freeze frames)")
    print("Competition : FIFA World Cup 2018")
    print("Strategy    : corner kick → linked shot → shot freeze frame")
    print("=" * 60)

    # Quick diagnostic: inspect one match's raw JSON
    print("\n[diag] Inspecting match 1 raw events ...")
    try:
        from statsbombpy import sb
        matches = sb.matches(competition_id=43, season_id=3)
        mid = int(matches["match_id"].iloc[0])

        evs = _fetch_raw_events(requests, mid)
        corners = [e for e in evs if e.get("type", {}).get("name") == "Pass"
                   and e.get("pass", {}).get("type", {}).get("name") == "Corner"]
        shots_ff = [e for e in evs if e.get("type", {}).get("name") == "Shot"
                    and isinstance(e.get("shot", {}).get("freeze_frame", None), list)
                    and len(e.get("shot", {}).get("freeze_frame", [])) > 0]

        print(f"  Match {mid}  |  corners: {len(corners)}  |  shots with freeze_frame: {len(shots_ff)}")

        if shots_ff:
            ff_sample = shots_ff[0].get("shot", {}).get("freeze_frame", [])
            print(f"  Sample freeze_frame entry: {ff_sample[0]}")

    except Exception as e:
        print(f"  [diag error] {e}")

    print()
    # Force re-fetch (ignore bad cache)
    sequences = load_corners(use_cache=False)

    if sequences:
        print_sample(sequences, n=3)
        from collections import Counter
        mc = Counter(s["match_id"] for s in sequences)
        print(f"\n  Matches with sequences : {len(mc)}")
        print(f"  Total sequences        : {len(sequences)}")
        print(f"  Avg sequences/match    : {len(sequences)/max(len(mc),1):.1f}")
        avg_p = sum(len(s["freeze_frame"]) for s in sequences) / len(sequences)
        print(f"  Avg players/frame      : {avg_p:.1f}")
    else:
        print("\n[FAILED] Still 0 sequences. Check internet access to github.com.")
        print("         Test with: python -c \"import requests; print(requests.get('https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json').status_code)\"")
