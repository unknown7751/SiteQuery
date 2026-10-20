# RAG evaluation report

- Generated: 2026-10-20 18:49 UTC
- Benchmark: 10 synthetic construction questions over 4 documents
- Generator / judge: `openai/gpt-oss-120b` (the model judges its own answers; expect optimistic bias)
- Embeddings: `BAAI/bge-small-en-v1.5` · Reranker: `Xenova/ms-marco-MiniLM-L-6-v2` (threshold 0.10) · top_k = 5

## Summary

| Variant | Faithfulness | Answer relevance | Context precision | Context hit@k | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid (RRF) only | 1.00 | 0.83 | 0.98 | 1.00 | 1.00 |
| Hybrid + cross-encoder rerank | 0.97 | 0.75 | 0.88 | 0.90 | 0.90 |

## Per-question: Hybrid (RRF) only

| ID | Faithfulness | Answer relevance | Context precision | Context hit@k | MRR | Question |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | What minimum 28-day compressive strength is required for elevated slabs and columns? |
| Q2 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | How many days must concrete be moist-cured? |
| Q3 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | How far ahead of the first concrete placement does the mix design have to be submitted? |
| Q4 | 1.00 | 0.70 | 1.00 | 1.00 | 1.00 | What is the status of RFI-1042, who must respond, and when is the response due? |
| Q5 | 1.00 | 0.65 | 1.00 | 1.00 | 1.00 | How was the curtain wall anchor embed conflict at gridline C/4 resolved? |
| Q6 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | Why was the structural steel delivery delayed and what is the new delivery date? |
| Q7 | 1.00 | 0.85 | 1.00 | 1.00 | 1.00 | Which ASTM standard and grade apply to wide-flange steel shapes, and what is the minimum yield strength? |
| Q8 | 1.00 | 0.86 | 1.00 | 1.00 | 1.00 | What does Change Order CO-07 cover and how much was approved? |
| Q9 | 1.00 | 0.78 | 1.00 | 1.00 | 1.00 | If it's freezing out when we pour, what do we have to do to protect the fresh concrete? |
| Q10 | 1.00 | 0.72 | 0.81 | 1.00 | 1.00 | Are we still waiting on the engineer about the duct opening clashing with a beam on the third floor? |

## Per-question: Hybrid + cross-encoder rerank

| ID | Faithfulness | Answer relevance | Context precision | Context hit@k | MRR | Question |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | What minimum 28-day compressive strength is required for elevated slabs and columns? |
| Q2 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | How many days must concrete be moist-cured? |
| Q3 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | How far ahead of the first concrete placement does the mix design have to be submitted? |
| Q4 | 1.00 | 0.76 | 0.83 | 1.00 | 1.00 | What is the status of RFI-1042, who must respond, and when is the response due? |
| Q5 | 1.00 | 0.61 | 1.00 | 1.00 | 1.00 | How was the curtain wall anchor embed conflict at gridline C/4 resolved? |
| Q6 | 1.00 | 0.81 | 1.00 | 1.00 | 1.00 | Why was the structural steel delivery delayed and what is the new delivery date? |
| Q7 | 1.00 | 0.85 | 1.00 | 1.00 | 1.00 | Which ASTM standard and grade apply to wide-flange steel shapes, and what is the minimum yield strength? |
| Q8 | 1.00 | 0.86 | 1.00 | 1.00 | 1.00 | What does Change Order CO-07 cover and how much was approved? |
| Q9 | 0.75 | 0.80 | 1.00 | 1.00 | 1.00 | If it's freezing out when we pour, what do we have to do to protect the fresh concrete? |
| Q10 | n/a | 0.00 | 0.00 | 0.00 | 0.00 | Are we still waiting on the engineer about the duct opening clashing with a beam on the third floor? |

## Answers (Hybrid + cross-encoder rerank)

**Q1. What minimum 28-day compressive strength is required for elevated slabs and columns?**
- Ground truth: Elevated slabs and columns require a minimum compressive strength of 5,000 psi at 28 days, with a maximum water-cementitious materials ratio of 0.42.
- Answer: The specification requires a **minimum 28‑day compressive strength of 5,000 psi** for elevated slabs and columns[1].

**Q2. How many days must concrete be moist-cured?**
- Ground truth: Concrete must be moist-cured for a minimum of 7 days, or a curing compound complying with ASTM C309 may be applied.
- Answer: The specification requires that concrete be moist‑cured for **a minimum of 7 days**[1].

**Q3. How far ahead of the first concrete placement does the mix design have to be submitted?**
- Ground truth: Proposed mix designs must be submitted at least 14 calendar days before the first placement.
- Answer: The mix designs must be submitted **at least 14 calendar days before the first concrete placement**[1].

**Q4. What is the status of RFI-1042, who must respond, and when is the response due?**
- Ground truth: RFI-1042 (Level 3 slab penetration conflict with beam B-14) is Open. Halvorsen & Lee, the Structural Engineer of Record, must respond by March 11, 2025.
- Answer: - **Status:** Open [1] - **Responder:** Halvorsen & Lee (Structural Engineer of Record) [1][3] - **Response due:** 2025‑03‑11 [1][3]

**Q5. How was the curtain wall anchor embed conflict at gridline C/4 resolved?**
- Ground truth: RFI-1043 was closed: shift the embed plates 3 inches north and add two #5 trim bars at each embed, per revised detail 7/A-502 issued with Bulletin 06.
- Answer: The conflict was resolved by **shifting the embed plates 3 inches north** and **adding two #5 trim bars at each embed**. A revised detail (7/A‑502) was issued with Bulletin 06 to reflect the change. [1]

**Q6. Why was the structural steel delivery delayed and what is the new delivery date?**
- Ground truth: The steel mill moved the rolling date for the wide-flange shapes, so delivery slipped from March 17 to March 24, 2025.
- Answer: - **Reason for delay:** The steel mill moved the rolling date for the wide‑flange shapes, causing the delivery slip. - **New delivery date:** March 24, 2025. [1]

**Q7. Which ASTM standard and grade apply to wide-flange steel shapes, and what is the minimum yield strength?**
- Ground truth: Wide-flange shapes must be ASTM A992, Grade 50, with a minimum yield strength of 50 ksi.
- Answer: - ASTM standard: **ASTM A992** - Grade: **Grade 50** - Minimum yield strength: **50 ksi** [1]

**Q8. What does Change Order CO-07 cover and how much was approved?**
- Ground truth: CO-07 for $48,250 was approved by the Owner and covers additional below-grade waterproofing at the elevator pit, including a second membrane layer and protection board.
- Answer: Change Order CO‑07 was approved for **$48,250** and it **covers additional below‑grade waterproofing at the elevator pit, including a second membrane layer and a protection board**[1].

**Q9. If it's freezing out when we pour, what do we have to do to protect the fresh concrete?**
- Ground truth: In cold weather, maintain the concrete temperature at or above 50 degrees F for the first 72 hours after placement. Placing below 40 degrees F requires an approved cold-weather plan.
- Answer: To place concrete when the temperature is at or below freezing you must: * **Obtain an approved cold‑weather plan** – concrete may not be placed when the ambient temperature is below 40 °F unless the Architect has approved a cold‑weather plan [1]. * **Keep the fresh concrete warm** – once placed, the concrete temperature must be maintained at **≥ 50 °F for the first 72 hours** [1]. These actions protect the fresh concrete from freezing damage.

**Q10. Are we still waiting on the engineer about the duct opening clashing with a beam on the third floor?**
- Ground truth: Yes. RFI-1042 (Level 3 mechanical riser penetration vs beam B-14) is still Open; Halvorsen & Lee's response is due March 11, 2025.
- Answer: I couldn't find any relevant passages in the selected documents.


## Metric definitions

- **Faithfulness**: share of answer claims supported by the retrieved context (LLM judge).
- **Answer relevance**: mean embedding cosine between the question and questions the judge infers from the answer. Scores are relative; compare variants rather than reading the absolute value.
- **Context precision**: average precision of the retrieved ranking against the ground truth (LLM judge).
- **Context hit@k / MRR**: deterministic check that the gold evidence string appears in the retrieved chunks, and the reciprocal rank of the first chunk that has it.
