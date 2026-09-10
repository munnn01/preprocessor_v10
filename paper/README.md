# Manuscript package

`manuscript.pdf` is the reviewable pre-results paper. `manuscript.md` is its authoritative editable text; `main.tex` and `references.bib` provide an IEEE-style submission source once real results and approved authorship are available.

Rebuild the PDF with:

```bash
python paper/build_pdf.py
```

Before any submission:

1. Run the locked protocol in `docs/EXPERIMENT_PROTOCOL.md`.
2. Populate both result tables only from archived per-video/Jetson artifacts.
3. Update `evidence/claims.csv` and `evidence/manuscript_manifest.json`.
4. Replace anonymous authors only after every contributor approves order, affiliations, acknowledgments, funding, and conflicts.
5. Perform human source, statistical, visual, and venue-format review.
