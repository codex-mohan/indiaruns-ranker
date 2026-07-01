"""Gradio sandbox for TalentLens."""

from pathlib import Path
import hashlib
import json
import queue
import threading
import logging
import os
import subprocess
import sys
import tempfile
import time

import gradio as gr
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sandbox")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SANDBOX_ARTIFACTS_ROOT = ROOT / "artifacts" / "sandbox"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_candidates.jsonl"
REQUIRED_ARTIFACTS = ("jd_emb.npy", "cand_embs.npy", "ids.npy", "tfidf.pkl", "manifest.json")
MAX_DEMO_CANDIDATES = int(os.getenv("MAX_DEMO_CANDIDATES", "100000"))
KEEPALIVE_SECONDS = 5.0


def _fmt_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _validate_demo_input(count: int) -> None:
    if count > MAX_DEMO_CANDIDATES:
        raise ValueError(
            f"Sandbox upload has more than {MAX_DEMO_CANDIDATES} candidates. "
            "The hosted HuggingFace demo is for small-sample reproducibility; "
            "full 100K ranking should be run locally with `python -m src.precompute` "
            "then `python -m src.rank`."
        )


def _count_jsonl(path: Path, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit is not None and count > limit:
                    return count
    return count


def _candidate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts_ready(artifacts_dir: Path, candidates_hash: str, candidate_count: int) -> bool:
    if any(not (artifacts_dir / name).exists() for name in REQUIRED_ARTIFACTS):
        return False
    try:
        manifest = json.loads((artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("candidate_count") == candidate_count
        and manifest.get("candidate_file_sha256") == candidates_hash
    )


def _run_command(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    label: str = "",
    progress=None,
    start: float = 0.0,
    end: float = 1.0,
) -> str:
    log.info("Starting: %s", label)
    log.info("  Command: %s", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    last_progress = t0

    while proc.poll() is None:
        now = time.perf_counter()
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            lines.append(item)
            if item:
                log.info("  [%s] %s", label, item)
        elapsed = now - t0
        if progress is not None and now - last_progress >= KEEPALIVE_SECONDS:
            fraction = min(end, start + (end - start) * min(elapsed / max(timeout, 1), 0.95))
            progress(fraction, desc=f"{label}: still running ({elapsed:.0f}s elapsed)")
            last_progress = now
        if elapsed > timeout:
            proc.kill()
            raise TimeoutError(f"{label} timed out after {timeout}s")
        time.sleep(0.5)

    return_code = proc.wait(timeout=5)
    reader.join(timeout=2)
    while True:
        try:
            item = output_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            continue
        lines.append(item)
        if item:
            log.info("  [%s] %s", label, item)

    elapsed = time.perf_counter() - t0
    log.info("Finished: %s in %.1fs (exit=%d)", label, elapsed, return_code)
    output = "\n".join(lines)
    if return_code != 0:
        raise RuntimeError(output[-2000:] if output else f"{label} failed (exit {return_code})")
    return output


def _run_ranker(candidates_path: Path, progress=gr.Progress()):
    out_path = Path(tempfile.gettempdir()) / "talentlens_ranked_output.csv"
    started = time.perf_counter()

    log.info("=== Sandbox run started ===")
    log.info("Candidates: %s", candidates_path)

    progress(0.05, desc=f"Inspecting candidate file (hosted demo limit: {MAX_DEMO_CANDIDATES} candidates)")

    size_bytes = candidates_path.stat().st_size
    count = _count_jsonl(candidates_path, limit=MAX_DEMO_CANDIDATES)
    log.info("Candidate count: %d%s", count, "+" if count > MAX_DEMO_CANDIDATES else "")
    log.info("Candidate file size: %s", _fmt_mb(size_bytes))
    _validate_demo_input(count)
    candidates_hash = _candidate_file_hash(candidates_path)
    artifacts_dir = SANDBOX_ARTIFACTS_ROOT / candidates_hash[:16]
    log.info("Artifacts dir: %s", artifacts_dir)
    log.info("Artifacts exist: %s", _artifacts_ready(artifacts_dir, candidates_hash, count))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT)

    was_cached = _artifacts_ready(artifacts_dir, candidates_hash, count)
    if not was_cached:
        log.info("Precompute needed — downloading models and building index")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        t_precompute = time.perf_counter()
        _run_command(
            [
                sys.executable,
                "-m",
                "src.precompute",
                "--candidates",
                str(candidates_path),
                "--artifacts",
                str(artifacts_dir),
            ],
            env,
            timeout=1800,
            label="precompute",
            progress=progress,
            start=0.15,
            end=0.70,
        )
        precompute_elapsed = time.perf_counter() - t_precompute
        log.info("Precompute total: %.1fs", precompute_elapsed)
        progress(0.70, desc=f"Precompute done in {precompute_elapsed:.0f}s — now ranking")
    else:
        log.info("Artifacts cached — skipping precompute")
        progress(0.70, desc="Artifacts cached — ranking candidates")
    log.info("Starting ranking")
    t_rank = time.perf_counter()
    progress(0.75, desc="Ranking candidates...")
    _run_command(
        [
            sys.executable,
            "-m",
            "src.rank",
            "--candidates",
            str(candidates_path),
            "--artifacts",
            str(artifacts_dir),
            "--out",
            str(out_path),
            "--allow-partial",
        ],
        env,
        timeout=600,
        label="rank",
        progress=progress,
        start=0.75,
        end=0.95,
    )
    rank_elapsed = time.perf_counter() - t_rank
    log.info("Ranking total: %.1fs", rank_elapsed)

    elapsed = time.perf_counter() - started
    log.info("=== Sandbox run complete in %.1fs ===", elapsed)

    progress(1.0, desc=f"Done in {elapsed:.0f}s")

    df = pd.read_csv(out_path)
    top_score = f"{df['score'].iloc[0]:.4f}" if len(df) else "n/a"

    # Build timing breakdown
    timing_parts = [f"ranking: {rank_elapsed:.1f}s"]
    if not was_cached:
        timing_parts.insert(0, f"precompute: {precompute_elapsed:.1f}s")
    timing_str = ", ".join(timing_parts)

    cached_note = "reused cached artifacts" if was_cached else "first run — artifacts now cached"

    metrics = (
        f"### Run complete\n"
        f"- Input candidates: **{count}**\n"
        f"- Ranked output rows: **{len(df)}**\n"
        f"- Top score: **{top_score}**\n"
        f"- Total time: **{elapsed:.1f}s** ({timing_str})\n"
        f"- Cache: {cached_note}\n"
        f"- Guardrail: hosted demo accepts up to **{MAX_DEMO_CANDIDATES} candidates**; full 100K runs locally via CLI because HF free runtime can drop long uploads/precompute jobs.\n\n"
        "Cached bundled sample runs in about **2.2s** on this Space. "
        "Local Gradio uses this same app; for full ranking, use the CLI path without `--allow-partial`."
    )
    return str(out_path), df.head(20), metrics


def rank_uploaded(candidate_file, progress=gr.Progress()):
    if candidate_file is None:
        return None, pd.DataFrame(), "### Upload required\nPlease upload a `.jsonl` candidate file."
    try:
        uploaded_path = Path(candidate_file.name)
        return _run_ranker(uploaded_path, progress)
    except Exception as exc:
        msg = str(exc)
        log.error("Run failed: %s", msg)
        return None, pd.DataFrame(), f"### Run failed\n```\n{msg[-2000:]}\n```"


def rank_sample(progress=gr.Progress()):
    try:
        if not SAMPLE_PATH.exists():
            return None, pd.DataFrame(), f"### Missing sample\n`{SAMPLE_PATH}` was not found."
        return _run_ranker(SAMPLE_PATH, progress)
    except Exception as exc:
        msg = str(exc)
        log.error("Run failed: %s", msg)
        return None, pd.DataFrame(), f"### Run failed\n```\n{msg[-2000:]}\n```"


THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Manrope"), "Inter", "Arial", "sans-serif"],
).set(
    body_background_fill="#080914",
    body_text_color="#f8fafc",
    block_background_fill="#111322",
    block_border_color="#2b2f45",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_hover="#8b5cf6",
    button_primary_text_color="#ffffff",
)

CSS = """
body {
  background: #080914 !important;
}
.gradio-container {
  width: min(100% - 64px, 1480px) !important;
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 36px 0 64px !important;
}
.hero {
  padding: 34px 36px;
  border-radius: 18px;
  background:
    radial-gradient(circle at 18% 18%, rgba(255, 94, 20, .45), transparent 28%),
    radial-gradient(circle at 85% 8%, rgba(124, 58, 237, .52), transparent 32%),
    linear-gradient(135deg, #0b1020 0%, #151225 48%, #090a12 100%);
  border: 1px solid rgba(255,255,255,.12);
  box-shadow: 0 24px 80px rgba(0,0,0,.32);
}
.hero,
.main-panel,
.output-panel {
  width: 100%;
}
.hero h1 {
  margin: 0 0 10px;
  font-size: 42px;
  line-height: 1.05;
  letter-spacing: 0;
}
.hero p {
  margin: 0;
  max-width: 780px;
  color: #cbd5e1;
  font-size: 16px;
}
.statbar {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.stat {
  padding: 14px 16px;
  border-radius: 12px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.12);
}
.stat strong {
  display: block;
  color: #fff;
  font-size: 20px;
}
.stat span {
  color: #cbd5e1;
  font-size: 12px;
}
.main-panel {
  margin-top: 28px;
}
.main-panel > div {
  gap: 22px;
}
.action-stack {
  align-self: stretch;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 18px;
}
.action-stack button {
  min-height: 54px !important;
  font-weight: 800 !important;
}
.output-panel {
  margin-top: 22px;
}
@media (max-width: 900px) {
  .gradio-container {
    width: min(100% - 28px, 1480px) !important;
    padding-top: 18px !important;
  }
  .hero {
    padding: 26px 22px;
  }
  .hero h1 {
    font-size: 34px;
  }
  .statbar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
footer { display: none !important; }
"""


with gr.Blocks(title="TalentLens") as app:
    gr.HTML(
        """
        <section class="hero">
          <h1>TalentLens</h1>
          <p>
            A deterministic hybrid ranking system for Redrob's Senior AI Engineer challenge.
            It combines retrieval semantics, lexical evidence, skill trust, career fit,
            behavioral availability, honeypot gating, and factual reasoning.
          </p>
          <div class="statbar">
            <div class="stat"><strong>100K</strong><span>full candidate pool</span></div>
            <div class="stat"><strong>160s</strong><span>latest full CPU rank step</span></div>
            <div class="stat"><strong>0/100</strong><span>gated honeypots in top 100</span></div>
            <div class="stat"><strong>No API</strong><span>offline rank step</span></div>
          </div>
        </section>
        """
    )

    with gr.Row(elem_classes=["main-panel"]):
        with gr.Column(scale=5):
            file_input = gr.File(label="Upload candidates JSONL", file_types=[".jsonl"])
        with gr.Column(scale=2, elem_classes=["action-stack"]):
            sample_btn = gr.Button("Run Bundled Sample", variant="secondary")
            run_btn = gr.Button("Rank Uploaded File", variant="primary")

    gr.Markdown(
        "### Hosted demo warning\n"
        "HuggingFace free/runtime may limit upload duration or drop long CPU-bound precompute jobs. "
        "The bundled sample is the reliable demo path; if a full 100K upload times out here, "
        "run the documented CLI workflow locally."
    )

    with gr.Column(elem_classes=["output-panel"]):
        metrics_output = gr.Markdown("### Ready\nUse the bundled sample or upload JSONL. First run auto-precomputes local artifacts, then ranks.")

        with gr.Row():
            csv_output = gr.File(label="Download ranked CSV")

        preview_output = gr.Dataframe(
            label="Top-ranked preview",
            interactive=False,
            wrap=True,
        )

    sample_btn.click(
        fn=rank_sample,
        inputs=[],
        outputs=[csv_output, preview_output, metrics_output],
    )
    run_btn.click(
        fn=rank_uploaded,
        inputs=[file_input],
        outputs=[csv_output, preview_output, metrics_output],
    )


if __name__ == "__main__":
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
        css=CSS,
    )
