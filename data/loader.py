"""
data/loader.py — Multi-competition StatsBomb loader (v2)
=========================================================
v2 change: loads from 6 free StatsBomb competitions instead of just WC2018.
Target: ~1,500 corner→shot sequences (vs 228 in v1).

Why more data matters:
    GCN v1 reached 11.8% val accuracy on 228 samples.
    With 6× more data, both models can generalise better.
    GAT benefits most — its 76k parameters need more examples
    to learn meaningful attention weights.

Competitions loaded (all free, no API key):
    ID 43 / Season   3  — FIFA World Cup 2018       (64 matches)
    ID 43 / Season 106  — FIFA World Cup 2022       (64 matches)
    ID 55 / Season  43  — UEFA Euro 2020            (51 matches)
    ID 11 / Season  90  — La Liga 2020/21           (38 matches)
    ID 11 / Season  42  — La Liga 2019/20           (38 matches)
    ID  2 / Season  44  — Premier League 2003/04    (38 matches)

Total: ~293 matches → target ~1,500 sequences.
"""

import pickle
import random
import warnings
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).parent
CACHE_FILE = DATA_DIR / "corners_cache_v2.pkl"   # separate cache from v1
GITHUB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SHOT_WINDOW_SEC = 12

# ── All competitions to load ──────────────────────────────────────────────────
# Format: (competition_id, season_id, label)
# These are all freely available from StatsBomb's open data GitHub.
COMPETITIONS = [
    (43,   3,  "FIFA World Cup 2018"),
    (43, 106,  "FIFA World Cup 2022"),
    (55,  43,  "UEFA Euro 2020"),
    (11,  90,  "La Liga 2020/21"),
    (11,  42,  "La Liga 2019/20"),
    ( 2,  44,  "Premier League 2003/04"),
]


def load_corners(use_cache: bool = True, max_matches_per_comp: int = None) -> list[dict]:
    """
    Load corner→shot sequences from multiple StatsBomb competitions.

    Parameters
    ----------
    use_cache             : load from disk if cache exists (skip re-downloading)
    max_matches_per_comp  : limit matches per competition (useful for quick testing)

    Returns
    -------
    List of dicts, each representing one corner kick that led to a shot within
    12 seconds. Keys: match_id, event_id, location, pass_end_location, team,
    freeze_frame, competition.
    """
    if use_cache and CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if len(data) > 0:
            print(f"[loader] {len(data)} sequences loaded from v2 cache.")
            return data
        print("[loader] Cache empty — re-fetching ...")

    try:
        from statsbombpy import sb
        import requests as req
    except ImportError as e:
        raise ImportError("Run: pip install statsbombpy requests") from e

    all_sequences = []

    for comp_id, season_id, label in COMPETITIONS:
        print(f"\n[loader] ── {label} (comp={comp_id}, season={season_id}) ──")

        try:
            matches_df = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            print(f"  [skip] Could not load match list: {e}")
            continue

        match_ids = matches_df["match_id"].tolist()
        if max_matches_per_comp:
            match_ids = match_ids[:max_matches_per_comp]

        comp_seqs = []
        for match_id in tqdm(match_ids, desc=f"  Loading", leave=False):
            try:
                seqs = _extract_from_match(req, int(match_id), comp_id, season_id)
                comp_seqs.extend(seqs)
            except Exception as exc:
                pass  # silently skip problem matches

        print(f"  → {len(comp_seqs)} sequences from {len(match_ids)} matches")
        all_sequences.extend(comp_seqs)

    print(f"\n[loader] Total: {len(all_sequences)} sequences across "
          f"{len(COMPETITIONS)} competitions.")

    if all_sequences:
        cache_to_disk(all_sequences)

    return all_sequences


def cache_to_disk(sequences: list[dict]) -> None:
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(sequences, f)
    print(f"[loader] Cached {len(sequences)} sequences → {CACHE_FILE}")


def _extract_from_match(req, match_id: int, comp_id: int, season_id: int) -> list[dict]:
    """Fetch raw events JSON and extract corner→shot sequences."""
    url = f"{GITHUB_RAW}/events/{match_id}.json"
    r = req.get(url, timeout=60)
    if r.status_code != 200:
        return []

    events = r.json()
    corners, shots_with_ff = [], []

    for ev in events:
        etype = ev.get("type", {}).get("name", "")
        if etype == "Pass":
            if ev.get("pass", {}).get("type", {}).get("name") == "Corner":
                if isinstance(ev.get("location"), list):
                    corners.append(ev)
        elif etype == "Shot":
            ff = ev.get("shot", {}).get("freeze_frame", [])
            if isinstance(ff, list) and len(ff) >= 4:
                shots_with_ff.append(ev)

    results = []
    for corner in corners:
        c_period  = corner.get("period")
        c_team_id = corner.get("team", {}).get("id")
        c_time    = corner.get("minute", 0) * 60 + corner.get("second", 0)
        c_loc     = corner.get("location", [])[:2]

        best_shot, best_gap = None, float("inf")
        for shot in shots_with_ff:
            if shot.get("period") != c_period:
                continue
            if shot.get("team", {}).get("id") != c_team_id:
                continue
            s_time = shot.get("minute", 0) * 60 + shot.get("second", 0)
            gap    = s_time - c_time
            if 0 < gap <= SHOT_WINDOW_SEC and gap < best_gap:
                best_gap, best_shot = gap, shot

        if not best_shot:
            continue

        ff           = best_shot.get("shot", {}).get("freeze_frame", [])
        shooter_loc  = best_shot.get("location", [])
        if len(shooter_loc) < 2:
            continue

        shooter_pos = best_shot.get("position", {"name": "Unknown"})
        if not isinstance(shooter_pos, dict):
            shooter_pos = {"name": "Unknown"}

        shooter_node = {
            "location" : shooter_loc[:2],
            "teammate" : True,
            "position" : shooter_pos,
            "actor"    : False,
            "player"   : best_shot.get("player", {"name": "Shooter"}),
        }
        insert_idx  = random.randint(0, len(ff))
        ff_extended = ff[:insert_idx] + [shooter_node] + ff[insert_idx:]

        results.append({
            "match_id"          : match_id,
            "event_id"          : str(best_shot.get("id", "")),
            "location"          : c_loc,
            "pass_end_location" : list(shooter_loc[:2]),
            "team"              : corner.get("team", {}).get("name", "Unknown"),
            "freeze_frame"      : ff_extended,
            "competition"       : f"{comp_id}/{season_id}",  # new field in v2
        })

    return results


def print_sample(sequences, n=2):
    for i, s in enumerate(sequences[:n]):
        print(f"\n--- Sequence {i+1} [{s.get('competition','?')}] ---")
        print(f"  team   : {s['team']}")
        print(f"  corner : {s['location']}  →  shooter at {s['pass_end_location']}")
        print(f"  players: {len(s['freeze_frame'])}")


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TacticAI-Lite v2 — Multi-Competition Loader")
    print("=" * 60)

    # Quick test: 5 matches per competition to verify connections fast
    seqs = load_corners(use_cache=False, max_matches_per_comp=5)
    print(f"\nQuick test (5 matches/comp): {len(seqs)} sequences")

    if seqs:
        print_sample(seqs, n=3)
        from collections import Counter
        comp_counts = Counter(s["competition"] for s in seqs)
        print("\nBreakdown by competition:")
        for comp, count in sorted(comp_counts.items()):
            print(f"  {comp:<10} : {count} sequences")