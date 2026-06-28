"""Check if generalization is required by the hackathon spec."""
import zipfile, xml.etree.ElementTree as ET

def read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml_content = z.read("word/document.xml")
    root = ET.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    texts = [t.text for t in root.findall(".//w:t", ns) if t.text]
    return " ".join(texts)

base = r"../data/India_runs_data_and_ai_challenge"
files = [("README", f"{base}/README.docx"), ("SUBMISSION_SPEC", f"{base}/submission_spec.docx")]

keywords = ["generaliz", "adapt", "other job", "multiple jd", "another role",
            "any jd", "different jd", "the jd", "a job description"]

for label, path in files:
    text = read_docx(path).lower()
    print(f"\n{'='*60}")
    print(f"Searching: {label}")
    print("="*60)
    found = False
    for kw in keywords:
        if kw in text:
            idx = text.index(kw)
            start = max(0, idx - 80)
            end = min(len(text), idx + 150)
            snippet = text[start:end]
            print(f'\n  Found "{kw}":')
            print(f'  ...{snippet}...')
            found = True
    if not found:
        print("  No generalization-related keywords found")

# Also check the JD's final note about the hackathon
print(f"\n{'='*60}")
print("JOB DESCRIPTION — final hackathon note")
print("="*60)
jd = read_docx(f"{base}/job_description.docx")
note_start = jd.lower().index("final note for the participants")
note = jd[note_start:note_start+800]
print(note)
