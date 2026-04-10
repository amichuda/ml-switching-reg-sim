# Academic Paper Writing Workflow

This project uses a multi-agent pipeline to research and write academic papers. Claude orchestrates specialized subagents — do not try to do everything in one context window.

## Project Structure

```
ml-switching-reg-sim/
│   └── README.md    
|   |__ analysis_files...            
└── paper-writer/            ← Claude Code agents live here
    ├── .claude/agents/      # subagent definitions
    ├── literature/          # output from literature-gatherer
    │   └── literature_review.md
    ├── outline/             # output from outline-planner + results-reader
    │   ├── paper_outline.md
    │   └── results_summary.md
    ├── draft/               # output from paper-writer / paper-reviser
    │   ├── paper_draft_v1.md
    │   └── paper_draft_v2.md
    ├── review/              # output from adversarial-critic / paper-reviser
    │   ├── referee_report.md
    │   ├── revision_checklist.md
    │   └── response_to_reviewer.md
    └── CLAUDE.md            # this file
```

## Agents

| Agent | Model | Role |
|---|---|---|
| `literature-gatherer` | Sonnet | Web search + Semantic Scholar source synthesis |
| `results-reader` | Sonnet | Reads analysis repo outputs → clean results digest |
| `outline-planner` | Sonnet | Argument structure + section map |
| `paper-writer` | Opus | Full draft authorship |
| `adversarial-critic` | Sonnet | Referee-style peer review |
| `paper-reviser` | Opus | Revision in response to critique |
| `analysis-coder` | Sonnet | Writes and runs code to generate new tables/figures |
| `pdf-formatter` | Sonnet | Renders QMD → PDF, checks JAE style, fixes formatting |

## Analysis Repo Path

```
ANALYSIS_REPO=../   # update this to match your actual layout
```

Tell Claude this path at the start of a session if it differs from the default, e.g.:
> "The analysis repo is at `~/projects/gig-economy-uganda`. Use results-reader on it."

## Standard Pipeline

Run agents in this order. Each depends on the previous agent's file output.

```
1. literature-gatherer  →  literature/literature_review.md
2. results-reader       →  outline/results_summary.md
3. outline-planner      →  outline/paper_outline.md
4. paper-writer         →  draft/paper_draft_v1.md
                           analysis_requests.md        ← logs any missing tables/figures
5. analysis-coder       →  ../analysis/scripts/*.do/R/py  (runs new code)
                           outline/results_summary.md  (appends new results)
                           analysis_requests.md        (marks requests complete)
6. adversarial-critic   →  review/referee_report.md
                           review/revision_checklist.md
                           analysis_requests.md        ← may add new requests
7. [analysis-coder]     →  fulfill any new requests from critic (optional)
8. paper-reviser        →  draft/paper_draft_v2.md
                           review/response_to_reviewer.md
9. pdf-formatter        →  draft/paper.qmd
                           draft/paper.pdf
                           draft/formatting_report.md
                           review/response_to_reviewer.md
```

Steps 1 and 2 are independent — run in either order or in parallel.
Step 5 is optional — only needed if paper-writer logged requests in `analysis_requests.md`.
Step 7 (pdf-formatter) runs last — after all prose revisions are done.
Repeat steps 6–8 as needed (typically 1–2 cycles).

## analysis_requests.md

This file is the communication channel between writing agents and the analysis-coder. The paper-writer and adversarial-critic log requests here when they need a table or figure that doesn't exist yet. Format:

```markdown
## Pending Requests

### [Descriptive name]
- **Requested by**: paper-writer
- **What's needed**: [Exact description — spec, sample, format]
- **Why**: [What argument in the paper it supports]
- **Status**: ⏳ Pending
```

Tell paper-writer to use this file explicitly:
> "If you need a table or figure that doesn't exist in results_summary.md, log it in analysis_requests.md instead of making up numbers."

## How to Start a New Paper

Tell Claude:

> "Start a new paper on [topic]. Use the literature-gatherer agent to build the literature review first, then we'll work through the pipeline."

Or to run the full pipeline automatically:

> "Run the full paper pipeline on [topic]. After the draft is complete, have the adversarial-critic review it, then use paper-reviser to produce a final version."

## How to Resume

If resuming a session mid-pipeline, tell Claude which files already exist and which agent to run next:

> "The literature review and outline are done. Use the paper-writer agent to write the draft."

## Cost Notes

- literature-gatherer and adversarial-critic use Sonnet (~$3/$15 per MTok) — run freely
- paper-writer and paper-reviser use Opus (~$5/$25 per MTok) — these are the expensive calls
- Use prompt caching if re-running the same literature context across multiple calls
- The Batch API gives 50% off if you're running non-interactive jobs

## Analysis Repo CLAUDE.md

If your analysis repo has its own `CLAUDE.md`, agents will read it automatically rather than re-exploring the repo from scratch. Keep it updated with:
- Output file locations and what each contains
- The main script execution order
- Any known data issues or caveats
- Variable naming conventions

The `results-reader` agent is specifically instructed to read it first. If you want other agents (e.g., `paper-writer`) to also reference it directly, say so when invoking them:
> "Use the paper-writer agent. Also read `../analysis/CLAUDE.md` for context on the data and identification strategy."

## Tips

- Invoke agents by name to be explicit: "Use the adversarial-critic agent to review draft v1."
- If the critic flags a fatal flaw in the methodology, address it before running paper-reviser — don't just cycle through blindly.
- You can re-run literature-gatherer with a narrower scope if the first pass is too broad.
- After 2 revision cycles, have the outline-planner re-read the final draft and check that the argument arc still holds.
- If your analysis repo changes (new results, corrected specs), re-run results-reader and then paper-reviser — do not edit the draft manually to reflect new numbers.
- The adversarial-critic reads `results_summary.md` alongside the draft, so it can catch cases where the draft misreports or overstates what the analysis actually shows.

## Quarto Skill and JAE Template

### Quarto Skill
Located at `.claude/skills/quarto/SKILL.md`. The `pdf-formatter` agent reads this automatically. It covers: rendering commands, YAML options, figure/table placement fixes, LaTeX error diagnosis, and post-render checks.

### JAE Template
Located at `templates/jae_template.qmd`. A fully-configured Quarto manuscript template with:
- Correct YAML header (12pt, double-spaced, 1" margins, natbib, jae.bst)
- LaTeX preamble (booktabs, fancyhdr, titlesec, placeins, caption)
- Section heading style (H1 = bold all-caps, H2/H3 = bold sentence case)
- Table and figure float environments with correct caption placement
- Appendix equation/table/figure numbering resets (A1, A2, ...)
- `\FloatBarrier` placeholders after each major section

**To use the template:** The `pdf-formatter` agent will scaffold `draft/paper.qmd` from this template automatically. Or copy it manually:
```bash
cp templates/jae_template.qmd draft/paper.qmd
```

### jae.bst
Download the JAE BibTeX style from CTAN and place in `draft/`:
```bash
cd draft && curl -O https://mirrors.ctan.org/biblio/bibtex/contrib/economic/jae.bst
```
