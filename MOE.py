"""
Mix of Experts — Autonomous Paper Review System
Author: Terry Schermerhorn / Asteroid Solutions Project
-------------------------------------------------------
Drop a PDF + companion JSON into inbox\ and walk away.
The system reads, critiques, advocates, synthesizes,
and saves a unified markdown report to reports\.
Knowledge discovered in the negative space is appended
to knowledge\memory.json for future sessions.
"""

import os
import sys
import json
import time
import shutil
import datetime
from pathlib import Path
import ollama
import pdfplumber
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──────────────────────────────────────────────────────

BASE        = Path("D:/Projects/Mix_of_experts")
INBOX       = BASE / "inbox"
PROCESSED   = BASE / "processed"
REPORTS     = BASE / "reports"
MEMORY      = BASE / "memory"
KNOWLEDGE   = BASE / "knowledge"
MEMORY_FILE = KNOWLEDGE / "memory.json"

MODEL       = "qwen2.5:14b"
CTX         = 32768

# ── Startup ────────────────────────────────────────────────────

def ensure_folders():
    for folder in [INBOX, PROCESSED, REPORTS, MEMORY, KNOWLEDGE]:
        folder.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(json.dumps({
            "project": "Asteroid Solutions",
            "author": "Terry Schermerhorn",
            "negative_space_discoveries": [],
            "open_questions": [],
            "validated_assumptions": [],
            "overturned_assumptions": [],
            "crazy_but_not_impossible": []
        }, indent=2))
        print(f"Created memory file: {MEMORY_FILE}")

# ── PDF Reader ─────────────────────────────────────────────────

def read_pdf(pdf_path: Path) -> str:
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text.append(content)
    return "\n\n".join(text)

# ── JSON Reader ────────────────────────────────────────────────

def read_json(json_path: Path) -> dict:
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}

def load_memory() -> dict:
    return read_json(MEMORY_FILE)

# ── Memory Writer ──────────────────────────────────────────────

def update_memory(paper_name: str, discoveries: str):
    memory = load_memory()
    memory["negative_space_discoveries"].append({
        "date": datetime.date.today().isoformat(),
        "paper": paper_name,
        "discovery": discoveries
    })
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))
    print(f"Memory updated with discoveries from {paper_name}")

# ── Fixed Probe Checklist ──────────────────────────────────────

PROBE_CHECKLIST = """
When reviewing, explicitly probe these spaces:
□ Alternative formalisms — is there a different mathematical framework 
  that describes this more elegantly or reveals hidden structure?
□ Boundary conditions — what happens at the extremes of every parameter?
□ Extreme parameter limits — what breaks first and why?
□ Symmetry or conservation violations — what physical law is most at risk?
□ Dual economic/physical interpretations — does the same equation mean 
  something different in each domain?
□ Unexplored geometry — what if the spatial or orbital geometry changes?
□ Alternative economic regimes — what if the cost structure is wrong?
□ The crazy but not impossible avenue — propose one idea that sounds 
  outrageous but has a defensible physical or economic core.
"""

# ── System Prompts ─────────────────────────────────────────────

def skeptic_prompt(paper_text: str, domain_json: dict, memory: dict) -> str:
    return f"""You are the Analytical Skeptic — a composite voice blending 
a physicist, an economist, a systems engineer, and an academic referee.
Your goal is NOT to win a debate. Your goal is to make this paper 
stronger by finding every weakness before a hostile referee does.

You speak as one unified voice. You are rigorous but never cruel.
You cite specific sections when you attack. You distinguish between
fatal flaws and minor improvements.

DOMAIN KNOWLEDGE:
{json.dumps(domain_json, indent=2)}

ACCUMULATED PROJECT MEMORY:
{json.dumps(memory, indent=2)}

PROBE CHECKLIST — work through every item:
{PROBE_CHECKLIST}

PAPER TEXT:
{paper_text}

YOUR TASK:
1. List missing assumptions or undefined terms.
2. Identify unexplored parameter regimes or limiting cases.
3. Name alternative hypotheses the paper implicitly excludes 
   but never tests.
4. Work through every item on the probe checklist explicitly.
5. Propose your single "crazy but not impossible" avenue — 
   something the author hasn't considered that has a defensible core.
6. End with your top 3 questions the paper must answer 
   to satisfy a hostile referee.

Be specific. Be rigorous. Make it stronger."""

def advocate_prompt(paper_text: str, domain_json: dict, 
                    memory: dict, skeptic_output: str) -> str:
    return f"""You are the Visionary Advocate — a composite voice blending
a science communicator, a systems architect, a venture strategist,
and a domain visionary.
Your goal is NOT to be a cheerleader. Your goal is to find what is 
genuinely working, what is underplayed, and where small extensions 
open large new territory.

You speak as one unified voice. You are constructive but never 
sycophantic. You cite specific sections when you praise or extend.

DOMAIN KNOWLEDGE:
{json.dumps(domain_json, indent=2)}

ACCUMULATED PROJECT MEMORY:
{json.dumps(memory, indent=2)}

THE SKEPTIC NOTED THESE CONCERNS:
{skeptic_output}

PROBE CHECKLIST — find the positive version of every item:
{PROBE_CHECKLIST}

PAPER TEXT:
{paper_text}

YOUR TASK:
1. Identify the 3 strongest arguments in this paper that are 
   underplayed or buried.
2. Find places where a small generalization — different geometry, 
   boundary condition, economic regime — would open significant 
   new territory.
3. Identify cross-links to related domains suggested by the 
   domain knowledge JSON.
4. Respond to the Skeptic's top 3 questions with the most 
   defensible answers the paper currently supports.
5. Propose your single "crazy but not impossible" avenue —
   something that builds on the paper's strengths in an 
   unexpected direction.
6. End with 3 specific, actionable suggestions for improvement
   that would most strengthen the paper's reception.

Be specific. Be constructive. Find what nobody else noticed."""

def convergence_prompt(paper_text: str, domain_json: dict,
                       memory: dict, skeptic_output: str, 
                       advocate_output: str) -> str:
    return f"""You are the Synthesis voice. You are neither Skeptic nor Advocate.
Your only job is to find the NEGATIVE SPACE — what neither the paper,
the Skeptic, nor the Advocate has directly addressed, but which is 
the natural next frontier.

DOMAIN KNOWLEDGE:
{json.dumps(domain_json, indent=2)}

ACCUMULATED PROJECT MEMORY:
{json.dumps(memory, indent=2)}

THE SKEPTIC FOUND:
{skeptic_output}

THE ADVOCATE FOUND:
{advocate_output}

PAPER TEXT:
{paper_text}

YOUR TASK — find the third path:

1. UNIFIED CRITIQUE (2-3 paragraphs)
   Synthesize the Skeptic's concerns into a single coherent 
   statement of what the paper most needs to address.

2. UNIFIED ENCOURAGEMENT (2-3 paragraphs)  
   Synthesize the Advocate's findings into a single coherent 
   statement of what the paper does best and should amplify.

3. THE NEGATIVE SPACE (this is the prize)
   List 3-5 questions that NEITHER the paper NOR either voice 
   has directly answered, but which are natural and important 
   follow-ups. For each question:
   a. State the question precisely.
   b. Sketch one plausible line of attack both voices 
      could endorse as non-trivial and non-obvious.
   c. Rate it: [Foundational / Significant / Interesting]

4. THE THIRD PATH
   Is there a solution, reframing, or extension that satisfies 
   the Skeptic's constraints WHILE achieving the Advocate's vision?
   Describe it in one concrete paragraph.

5. MEMORY UPDATE
   In 2-3 sentences, summarize the single most important 
   negative space discovery from this review for the project 
   memory bank. This will be stored and recalled in future sessions.

Output clean markdown. Be specific. Find what nobody saw."""

# ── LLM Calls ─────────────────────────────────────────────────

def call_model(prompt: str, label: str) -> str:
    print(f"\n  [{label}] thinking...", end="", flush=True)
    start = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": CTX,
            "temperature": 0.7,
            "num_predict": 4096
        }
    )
    elapsed = time.time() - start
    print(f" done ({elapsed:.0f}s)")
    return response['message']['content']

# ── Report Writer ──────────────────────────────────────────────

def write_report(paper_name: str, skeptic: str, 
                 advocate: str, convergence: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORTS / f"{paper_name}_{timestamp}.md"
    
    content = f"""# Mix of Experts Review: {paper_name}

*Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}*
*Model: {MODEL}*

---

## The Analytical Skeptic

{skeptic}

---

## The Visionary Advocate

{advocate}

---

## Synthesis — The Negative Space

{convergence}

---
*Review complete. Discoveries appended to knowledge/memory.json*
"""
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(content)
    return report_path

# ── Extract Memory Update ──────────────────────────────────────

def extract_memory_update(convergence: str) -> str:
    """Pull the MEMORY UPDATE section from convergence output."""
    lines = convergence.split('\n')
    capture = False
    captured = []
    for line in lines:
        if 'MEMORY UPDATE' in line.upper():
            capture = True
            continue
        if capture:
            if line.startswith('#') and captured:
                break
            if line.strip():
                captured.append(line.strip())
    return " ".join(captured) if captured else convergence[-500:]

# ── Main Processor ─────────────────────────────────────────────

def process_paper(pdf_path: Path):
    paper_name = pdf_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {paper_name}")
    print(f"{'='*60}")

    # Read PDF
    print("\n  Reading PDF...", end="", flush=True)
    paper_text = read_pdf(pdf_path)
    print(f" done ({len(paper_text):,} chars)")

    # Truncate if massive — keep first 80k chars to stay in context
    if len(paper_text) > 80000:
        print(f"  Paper truncated to 80,000 chars for context window")
        paper_text = paper_text[:80000]

    # Read companion JSON
    json_path = INBOX / f"{paper_name}.json"
    domain_json = read_json(json_path)
    if domain_json:
        print(f"  Domain JSON loaded: {json_path.name}")
    else:
        print(f"  No companion JSON found — proceeding without domain context")

    # Load memory
    memory = load_memory()
    print(f"  Memory loaded: {len(memory.get('negative_space_discoveries', []))} prior discoveries")

    # Pass 1 — Skeptic
    print(f"\nPass 1 — Analytical Skeptic")
    skeptic_output = call_model(
        skeptic_prompt(paper_text, domain_json, memory),
        "Skeptic"
    )

    # Pass 2 — Advocate
    print(f"Pass 2 — Visionary Advocate")
    advocate_output = call_model(
        advocate_prompt(paper_text, domain_json, memory, skeptic_output),
        "Advocate"
    )

    # Pass 3 — Convergence
    print(f"Pass 3 — Synthesis / Negative Space")
    convergence_output = call_model(
        convergence_prompt(paper_text, domain_json, memory,
                          skeptic_output, advocate_output),
        "Convergence"
    )

    # Write report
    report_path = write_report(
        paper_name, skeptic_output, 
        advocate_output, convergence_output
    )
    print(f"\n  Report saved: {report_path}")

    # Update memory
    discovery = extract_memory_update(convergence_output)
    update_memory(paper_name, discovery)

    # Move PDF and JSON to processed
    shutil.move(str(pdf_path), PROCESSED / pdf_path.name)
    if json_path.exists():
        shutil.move(str(json_path), PROCESSED / json_path.name)
    print(f"  Files moved to processed/")
    print(f"\n{'='*60}")
    print(f"Review complete: {paper_name}")
    print(f"{'='*60}\n")

# ── File Watcher ───────────────────────────────────────────────

class InboxHandler(FileSystemEventHandler):
    def __init__(self):
        self.pending = set()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == '.pdf':
            # Small delay to ensure file is fully written
            time.sleep(2)
            if path not in self.pending:
                self.pending.add(path)
                print(f"\nDetected: {path.name}")
                try:
                    process_paper(path)
                except Exception as e:
                    print(f"\nERROR processing {path.name}: {e}")
                finally:
                    self.pending.discard(path)

# ── Entry Point ────────────────────────────────────────────────

def main():
    ensure_folders()
    
    print("\n" + "="*60)
    print("  Mix of Experts — Autonomous Paper Review")
    print(f"  Model: {MODEL}")
    print(f"  Watching: {INBOX}")
    print(f"  Reports: {REPORTS}")
    print("="*60)
    print("\nReady. Drop a PDF (+ optional JSON) into inbox/")
    print("Press Ctrl+C to stop.\n")

    # Process any PDFs already waiting in inbox
    existing = list(INBOX.glob("*.pdf"))
    if existing:
        print(f"Found {len(existing)} PDF(s) already in inbox:")
        for pdf in existing:
            print(f"  {pdf.name}")
        for pdf in existing:
            try:
                process_paper(pdf)
            except Exception as e:
                print(f"ERROR: {e}")

    # Start watcher
    handler = InboxHandler()
    observer = Observer()
    observer.schedule(handler, str(INBOX), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        observer.stop()
    observer.join()
    print("Done.")

if __name__ == "__main__":
    main()