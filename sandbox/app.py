"""Gradio sandbox for the INDIA RUNS candidate ranker."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

import gradio as gr
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS_DIR = ROOT / "artifacts"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_candidates.jsonl"


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _run_ranker(candidates_path: Path, progress=gr.Progress()):
    out_path = Path(tempfile.gettempdir()) / "indiaruns_ranked_output.csv"
    progress(0.1, desc="Loading candidate profiles")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        sys.executable,
        "-m",
        "src.rank",
        "--candidates",
        str(candidates_path),
        "--artifacts",
        str(ARTIFACTS_DIR),
        "--out",
        str(out_path),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Unknown ranking error").strip()
        # Sanitize: remove absolute paths, keep error message
        lines = detail.splitlines()
        safe_lines = []
        for line in lines[-20:]:  # last 20 lines only
            if "File " in line and ("site-packages" in line or ROOT.name in line):
                continue  # skip internal tracebacks
            safe_lines.append(line)
        safe_detail = "\n".join(safe_lines) if safe_lines else "Ranking failed. Check input format."
        raise RuntimeError(safe_detail[-500:])

    progress(0.9, desc="Preparing preview")

    df = pd.read_csv(out_path)
    count = _count_jsonl(candidates_path)
    metrics = (
        f"### Run complete\n"
        f"- Input candidates: **{count}**\n"
        f"- Ranked output rows: **{len(df)}**\n"
        f"- Top score: **{df['score'].iloc[0]:.4f}**\n"
        f"- Time taken: **{elapsed:.1f}s**\n"
        f"- Output: `{out_path}`\n\n"
        "The sandbox uses the same deterministic ranking code as the full submission. "
        "For demo speed, use the bundled sample or upload a small JSONL file."
    )
    return str(out_path), df.head(20), metrics


def rank_uploaded(candidate_file, progress=gr.Progress()):
    if candidate_file is None:
        return None, pd.DataFrame(), "### Upload required\nPlease upload a `.jsonl` candidate file."
    try:
        uploaded_path = Path(candidate_file.name)
        return _run_ranker(uploaded_path, progress)
    except Exception as exc:
        msg = str(exc)[:200]
        return None, pd.DataFrame(), f"### Run failed\n`{msg}`"


def rank_sample(progress=gr.Progress()):
    try:
        if not SAMPLE_PATH.exists():
            return None, pd.DataFrame(), f"### Missing sample\n`{SAMPLE_PATH}` was not found."
        return _run_ranker(SAMPLE_PATH, progress)
    except Exception as exc:
        msg = str(exc)[:200]
        return None, pd.DataFrame(), f"### Run failed\n`{msg}`"


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


with gr.Blocks(title="INDIA RUNS Ranker") as app:
    gr.HTML(
        """
        <section class="hero">
          <h1>INDIA RUNS Candidate Ranker</h1>
          <p>
            A deterministic hybrid ranking system for Redrob's Senior AI Engineer challenge.
            It combines retrieval semantics, lexical evidence, skill trust, career fit,
            behavioral availability, honeypot gating, and factual reasoning.
          </p>
          <div class="statbar">
            <div class="stat"><strong>100K</strong><span>full candidate pool</span></div>
            <div class="stat"><strong>173s</strong><span>latest full CPU run</span></div>
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

    with gr.Column(elem_classes=["output-panel"]):
        metrics_output = gr.Markdown("### Ready\nUse the bundled sample for the fastest demo path.")

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
    app.launch(theme=THEME, css=CSS)
