"""Build deck.html from deck.md — print to PDF from browser."""
import markdown, re, os

with open("docs/deck.md", "r", encoding="utf-8") as f:
    md = f.read()

# Split on H1 (slide titles)
slides = re.split(r"(?=^# .+)", md, flags=re.MULTILINE)
slides_html = []
for slide in slides:
    slide = slide.strip()
    if not slide:
        continue
    html = markdown.markdown(slide, extensions=["tables", "fenced_code"])
    slides_html.append(f'<div class="slide">{html}</div>')

css = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; }
.slide {
    width: 960px; min-height: 540px; margin: 30px auto; padding: 40px 60px;
    background: #fff; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    page-break-after: always; page-break-inside: avoid;
}
.slide h1 { font-size: 24px; color: #2563eb; border-bottom: 3px solid #2563eb; padding-bottom: 10px; margin-bottom: 20px; }
.slide h2 { font-size: 20px; color: #1e40af; margin-top: 24px; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid #e5e7eb; }
.slide h3 { font-size: 16px; color: #374151; margin-top: 16px; margin-bottom: 6px; }
.slide p, .slide li { font-size: 14px; line-height: 1.7; color: #1f2937; }
.slide ul, .slide ol { margin-left: 20px; margin-top: 6px; margin-bottom: 10px; }
.slide li { margin-bottom: 4px; }
.slide code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-family: Consolas, monospace; font-size: 13px; color: #dc2626; }
.slide pre { background: #f3f4f6; padding: 14px; border-left: 3px solid #2563eb; margin: 10px 0; overflow-x: auto; font-size: 12px; }
.slide pre code { background: none; padding: 0; color: #1f2937; }
.slide strong { color: #1e40af; }
.slide table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
.slide th { background: #e8f0fe; padding: 8px 10px; text-align: left; border: 1px solid #cbd5e1; font-weight: bold; }
.slide td { padding: 7px 10px; border: 1px solid #e5e7eb; }
.slide hr { margin: 20px 0; border: none; border-top: 1px solid #e5e7eb; }
.slide blockquote { border-left: 4px solid #2563eb; padding: 8px 16px; margin: 10px 0; background: #f0f4ff; font-style: italic; }
@media print {
    body { background: #fff; }
    .slide { box-shadow: none; margin: 0; border-radius: 0; width: 100%; min-height: auto; }
}
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>INDIA RUNS — Candidate Ranking System</title><style>{css}</style></head>
<body>
{''.join(slides_html)}
</body></html>"""

with open("docs/deck.html", "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize("docs/deck.html") / 1024
print(f"HTML generated: docs/deck.html ({size_kb:.0f} KB)")
print("Open in browser and Ctrl+P -> Save as PDF")
