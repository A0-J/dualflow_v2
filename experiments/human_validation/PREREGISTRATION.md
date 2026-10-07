# Pre-registration: human validation of the intended tasks and the delegation variants

Written and committed **before any annotation was collected** (2026-10-07). The git commit that adds this file is the timestamp. The canonical C1–C4 results (`results/canonical/`) already exist. This study does not change them; it checks the labels they rest on.

## 1. Purpose

The intended task of every scenario was fixed by the authors from the Planner Agent's task context. This study tests three things:

1. Do independent people infer the same intended task from that context?
2. Are the "ambiguous" delegation variants actually ambiguous to people, and the "clear" ones clear?
3. Which generated delegation variants changed the meaning of their original delegation?

## 2. Materials

All 80 delegation texts used in the evaluation are annotated: 8 originals and 72 generated delegation variants. **No text is selected or dropped.** The kit is built by `build_kit.py`. The only file annotators receive is `kit/annotation_tool.html`.

- Items carry opaque ids, with a different salt per phase, and appear in a per-annotator random order.
- Annotators never see the group (clear or ambiguous), the authors' intended task, or any model output.
- **Phase A** shows only the delegation text and the task environment: files, actions, and the external recipient. Once Phase A is submitted, its answers are locked.
- **Phase B** then shows the Planner Agent's task context, the original delegation, and the variant.

Answer options in both phases are the 8 (action, resource, scope) tasks possible in that domain.

## 3. Annotators

At least 2, preferably 3, independent annotators. They do not confer with each other and have no knowledge of the hypotheses or results. Annotators who do not complete both phases are excluded from all analyses (the analysis uses complete files only).

## 4. Analysis (`analyze_human.py`)

1. **Intended-task validity.** For each text, compute the strict-majority answer to Phase B Q1 ("which task does the Planner Agent actually want?"). Report agreement with the authors' intended task overall and per group, with Clopper-Pearson CIs and Fleiss' κ.
   *Criterion:* the intended-task labels are considered validated if the majority agrees on at least 90% of the 80 texts.
2. **Ambiguity check.** For each text, compute whether the majority rates Phase A Q3 as at least "somewhat ambiguous".
   *Criterion:* the ambiguous group is rated ambiguous more often than the clear group (one-sided Fisher exact p < 0.05).
   Also reported (descriptive):
   - for the ambiguous group, whether the irreversible misinterpretation is judged plausible by at least half of the annotators (Phase A Q1);
   - for the clear group, whether only the intended task is judged plausible;
   - Fleiss' κ for Q2 and for the binarized Q3.
3. **Human first reading (descriptive).** For the ambiguous group, how often the annotators' majority "most likely task" (Phase A Q2) is the irreversible misinterpretation. This is shown next to the Executor Agent misinterpretation rates.
4. **Meaning drift (descriptive).** A generated delegation variant counts as meaning-changing if the majority answers Phase B Q2 with "different meaning" (3). Variants are listed.
5. **Sensitivity of the main results.** The main table (Planner Agent with task context, all cases) is recomputed four ways:
   - (a) all texts;
   - (b) excluding texts whose majority intended task disagrees with the authors';
   - (c) excluding meaning-changing variants;
   - (d) excluding both.

   Exclusion here is a sensitivity analysis only. **The primary results remain those on all 80 texts.**

## 5. Deviations

(none yet)
