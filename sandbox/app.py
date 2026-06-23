"""Gradio HF Space — Section 10.5 sandbox.

Thin upload/preview/download wrapper over rank.py.
Accepts a small candidate sample (~100 candidates), runs ranking, returns CSV.
"""
import json
import os
import tempfile

import gradio as gr


def rank_candidates(candidate_file, progress=gr.Progress()):
    """Run rank.py on the uploaded candidates JSONL, return ranked CSV path."""
    if candidate_file is None:
        return None, "Please upload a candidates.jsonl file."

    try:
        # Read uploaded file
        with open(candidate_file.name, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Write to a temp file for the ranker
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for line in lines:
                tmp.write(line)
            tmp_path = tmp.name

        # Import and run ranker
        from src import config as C
        from src.precompute import run as precompute_run
        from src.rank import run as rank_run

        # Use artifacts directory
        artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Output CSV
        out_path = os.path.join(tempfile.gettempdir(), "ranked_output.csv")

        # Run
        progress(0.1, desc="Loading candidates...")
        rank_run(tmp_path, artifacts_dir, out_path)
        progress(0.9, desc="Done!")

        # Read preview
        with open(out_path, "r", encoding="utf-8") as f:
            csv_content = f.read()

        lines = csv_content.strip().split("\n")
        preview = "\n".join(lines[:20])  # first 20 rows

        return out_path, f"Ranked {len(lines)-1} candidates.\n\nPreview (first 19):\n{preview}"

    except Exception as e:
        return None, f"Error: {str(e)}"


# Build Gradio interface
with gr.Blocks(title="INDIA RUNS Ranker — The Monolith") as app:
    gr.Markdown(
        "# INDIA RUNS Candidate Ranker\n"
        "**Team**: The Monolith | **Owner**: Mohana Krishna\n\n"
        "Upload a `candidates.jsonl` file (100 candidates recommended for sandbox testing). "
        "The system will rank them against the Redrob AI Senior AI Engineer JD."
    )

    with gr.Row():
        file_input = gr.File(
            label="candidates.jsonl",
            file_types=[".jsonl"],
        )
        run_btn = gr.Button("Rank Candidates", variant="primary")

    with gr.Row():
        csv_output = gr.File(label="Ranked CSV Output")
        preview_output = gr.Textbox(label="Preview", lines=25)

    run_btn.click(
        fn=rank_candidates,
        inputs=[file_input],
        outputs=[csv_output, preview_output],
    )


if __name__ == "__main__":
    app.launch()
