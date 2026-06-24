"""Compare old (bi-encoder only) vs new (cross-encoder) ranking."""
import csv, subprocess, os, tempfile

# Extract old CSV from git
repo = r"C:\Users\wwwmo\Development\Competitions\IndiaRuns\indiaruns-ranker"
old_path = os.path.join(tempfile.gettempdir(), "old_ranking.csv")
result = subprocess.run(
    ["git", "show", "6095b07:codexmohan_6487.csv"],
    capture_output=True, text=True, cwd=repo
)
with open(old_path, "w", encoding="utf-8") as f:
    f.write(result.stdout)

def load_csv(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["candidate_id"]] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"],
            }
    return data

old = load_csv(old_path)
new = load_csv(os.path.join(repo, "codexmohan_6487.csv"))

old_ids = set(old.keys())
new_ids = set(new.keys())
overlap = old_ids & new_ids
dropped = old_ids - new_ids
promoted = new_ids - old_ids

D = "=" * 95

print(D)
print("SIDE-BY-SIDE: Old (bi-encoder only) vs New (cross-encoder re-ranked)")
print("  CV candidate CAND_0068410 was rank 99 in old -> REMOVED in new")
print(D)
print()

old_by_rank = sorted(old.items(), key=lambda x: x[1]["rank"])
new_by_rank = sorted(new.items(), key=lambda x: x[1]["rank"])

for i in range(30):
    oid, od = old_by_rank[i]
    nid, nd = new_by_rank[i]
    otitle = od["reasoning"][:60]
    ntitle = nd["reasoning"][:60]

    moved = ""
    if oid in new:
        new_rank = new[oid]["rank"]
        if new_rank > i + 1:
            moved = f" [DOWN to {new_rank}]"
        elif new_rank < i + 1:
            moved = f" [UP to {new_rank}]"
    else:
        moved = " [DROPPED]"

    new_moved = ""
    if nid in old:
        old_rank = old[nid]["rank"]
        if old_rank > i + 1:
            new_moved = f" [was {old_rank}]"

    print(f" R{i+1:>2} | OLD {oid} {od['score']:.4f} {moved}")
    print(f"     |     {otitle}")
    print(f" R{i+1:>2} | NEW {nid} {nd['score']:.4f} {new_moved}")
    print(f"     |     {ntitle}")
    print()

print(D)
print(f"DROPPED from top 100 ({len(dropped)} candidates)")
print(D)
for oid in sorted(dropped):
    od = old[oid]
    print(f"  old rank {od['rank']:>3}: {oid} | {od['reasoning'][:90]}")

print()
print(D)
print(f"PROMOTED into top 100 ({len(promoted)} candidates)")
print(D)
for nid in sorted(promoted):
    nd = new[nid]
    print(f"  new rank {nd['rank']:>3}: {nid} | {nd['reasoning'][:90]}")

print()
print(D)
print("MOVEMENT SUMMARY")
print(D)
up = sum(1 for oid in overlap if new[oid]["rank"] < old[oid]["rank"])
down = sum(1 for oid in overlap if new[oid]["rank"] > old[oid]["rank"])
same = sum(1 for oid in overlap if new[oid]["rank"] == old[oid]["rank"])
big_movers = []
for oid in overlap:
    delta = old[oid]["rank"] - new[oid]["rank"]
    if abs(delta) >= 10:
        big_movers.append((oid, old[oid]["rank"], new[oid]["rank"], delta, old[oid]["reasoning"][:80]))
big_movers.sort(key=lambda x: -abs(x[3]))
print(f"  Overlap:  {len(overlap)}/100")
print(f"  Moved UP:   {up}")
print(f"  Moved DOWN: {down}")
print(f"  Same rank:  {same}")
print(f"  Dropped:    {len(dropped)}")
print(f"  Promoted:   {len(promoted)}")
if big_movers:
    print(f"\n  Biggest movers (>=10 ranks):")
    for oid, orank, nrank, delta, reason in big_movers[:10]:
        direction = "UP" if delta > 0 else "DOWN"
        print(f"    {oid}: {orank} -> {nrank} ({direction} {abs(delta)}) | {reason}")
