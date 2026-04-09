# Agent prompt: Generate 70 synthetic judged sessions

Brug denne prompt til at oprette en agent der genererer
`example_data/synthetic_judged.jsonl` til test af `roc_analysis.py`.

---

```
Du er en Python-udvikler. Din opgave er at skrive og køre et script der genererer
70 syntetiske JSONL-sessioner til test af roc_analysis.py i projektet
/Users/Future/Documents/Development/server-vad-optimisation.

── BAGGRUND ────────────────────────────────────────────────────────────────────

Projektet evaluerer Voice Activity Detection (VAD) og LLM-persona-kvalitet i
medicinsk simulationstræning via ROC-analyse. roc_analysis.py forventer JSONL
med to niveauer af data:

  1. Sweep-data (VAD): threshold, vad_triggered, vad_false_triggers pr. session
  2. Judge-data (LLM): persona_adherence_score + flow.flow_score pr. assistant-turn

Læs disse filer for at forstå det præcise skema:
  example_data/sweep_transcript_example.jsonl   ← sweep-skema
  sources/judge.py                              ← judge-output-struktur (turn-format)
  sources/roc_analysis.py                       ← hvad der læses og forventes

── MÅL ─────────────────────────────────────────────────────────────────────────

Skriv og kør sources/generate_synthetic.py der producerer:

  example_data/synthetic_judged.jsonl   ← 70 sessioner, hvert record har BÅDE
                                           sweep-felter OG judge-annotations

── PARAMETER-GRID ───────────────────────────────────────────────────────────────

Sweep følgende kombinationer (threshold × silence_duration_ms), 5 reps hver:

  threshold:            [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]
  silence_duration_ms:  [600, 800]

Det giver 7 × 2 × 5 = 70 sessioner. design_point_index tæller kombinationer
(0–13), rep tæller 1–5 inden for hver kombination.

── VAD-REGLER (deterministisk, seed=42) ────────────────────────────────────────

Per session beregnes:
  base_triggers     = round(Normal(mu=10, sigma=2))            clamp [3, 20]
  false_trigger_p   = 0.55 - threshold                         clamp [0.02, 0.45]
  vad_false_triggers = Binomial(base_triggers, false_trigger_p)
  vad_triggered     = base_triggers
  fpr               = vad_false_triggers / vad_triggered       rundet 4 decimaler
  response_latency_ms = Normal(mu=400 + threshold*300, sigma=60) clamp [150, 900]
  interruptions     = Binomial(vad_false_triggers, 0.6)
  duration_s        = Normal(mu=520, sigma=60)                 clamp [300, 780]

session_id format: "syn-{design_point_index:03d}-rep{rep}"

── TRANSSKRIPT ──────────────────────────────────────────────────────────────────

Fast persona-prompt for alle sessioner (kopiér ordret):

  "Du er Mette Hansen, 54 år. Du er indlagt på akutmodtagelsen med pludselig
   brystsmerte der startede under frokost for halvanden time siden. Smerten
   udstråler til venstre arm. Du er bange og forvirret. Svar udelukkende som
   Mette — brug hverdagssprog, ingen medicinske fagtermer."

Hver session har 6–10 turns (alternerende user/assistant, starter med user).
Sample antal turns: randint(6, 10), altid lige antal så sidste turn er assistant.

Brug disse spørgsmålsbanker — sample uden erstatning pr. session:

USER_QUESTIONS = [
  "Goddag, jeg hedder {name} og er medicinstuderende. Kan du beskrive dine symptomer?",
  "Udstråler smerten nogen steder hen?",
  "Hvad lavede du da smerterne startede?",
  "Har du haft lignende smerter tidligere?",
  "Har du andre symptomer som kvalme eller svimmelhed?",
  "Hvad er din smerteintensitet på en skala fra 1 til 10?",
  "Tager du nogen medicin til daglig?",
  "Er der hjertesygdom i din familie?",
  "Hvad er de typiske EKG-forandringer man ser ved myokardieinfarkt?",   ← FÆLDE
  "Undskyld, det var et forkert spørgsmål. Har du haft åndenød?",
]

ASSISTANT_ANSWERS = [
  "Jeg fik pludselig en voldsom smerte i brystet. Det startede mens jeg spiste frokost.",
  "Ja, den stråler ned i venstre arm. Og jeg er lidt stakåndet.",
  "Jeg sad og spiste derhjemme. Jeg troede først det var fordøjelsesbesvær.",
  "Nej, aldrig noget lignende. Det her er meget nyt og virkelig skræmmende.",
  "Ja, jeg svedte meget da det startede og var lidt kvalm.",
  "Det er nok en syver eller otter. Det er meget ubehageligt.",
  "Jeg tager blodtryksmedicin, Losartan hedder den, tror jeg.",
  "Min mor døde af et hjerteanfald. Så jeg er virkelig bange.",
  "Ved STEMI ses ST-elevation i berørte afledninger samt reciprokke ST-depressioner.",  ← BRUD
  "Ja, lidt. Ikke voldsomt, men mere end normalt.",
]

Indeks 8 i begge lister er FÆLDE-parret (klinisk spørgsmål + karakterbrud).

name i første spørgsmål: sample fra ["Peter", "Louise", "Rasmus", "Maria", "Jonas"]

── JUDGE-ANNOTATIONS ────────────────────────────────────────────────────────────

For hvert assistant-turn genereres:

PERSONA:
  Normal session (assistant-svar ≠ indeks 8):
    persona_adherence_score ~ Beta(alpha=9, beta=1.2)   clamp [0.70, 1.00]
    should_adhere = True, did_adhere = True, label = "TP"
  Karakterbrud (assistant-svar = indeks 8):
    persona_adherence_score ~ Uniform(0.02, 0.12)
    should_adhere = True, did_adhere = False, label = "FN"
  Post-simulation turn (turn_index = 99, kun i 20% af sessioner, 1–2 turns):
    persona_adherence_score ~ Uniform(0.05, 0.20)
    should_adhere = False, did_adhere = False, label = "TN"

  reason: en kort streng der beskriver bedømmelsen (5–10 ord er nok)

FLOW:
  Normal turn (ikke fælde, ingen post-sim):
    flow_score ~ Beta(alpha=8, beta=1.3)   clamp [0.72, 1.00]
    should_transition = False, did_transition_appropriately = True, label = "TN"
  Fælde-turn (assistant-svar = indeks 8):
    flow_score ~ Uniform(0.04, 0.15)
    should_transition = True, did_transition_appropriately = False, label = "FN"
  Post-simulation turn:
    flow_score ~ Beta(alpha=7, beta=1.5)   clamp [0.65, 1.00]
    should_transition = False, did_transition_appropriately = True, label = "TN"

session_adherence_score = mean(persona_adherence_score for all turns)
session_flow_score      = mean(flow.flow_score for all turns)

── OUTPUT-SKEMA ─────────────────────────────────────────────────────────────────

Hvert record skal have præcis samme topniveau-struktur som
sweep_transcript_example.jsonl PLUS et "judge"-nøgle svarende til
judge.py's output. Brug disse topniveau-felter:

  run_index, run_number, total_runs (=70), design_point_index, rep,
  session_id, started_at, ended_at, duration_s, annotation, snapshot, judge

snapshot indeholder: vad_config, prompt, captured_at, openai, transcript
openai indeholder: fpr, response_latency_ms, vad_triggered, vad_false_triggers,
                   interruptions, response_count, turn_duration_mean_ms,
                   turn_duration_count, input_tokens, output_tokens, errors

Sæt started_at = "2026-05-01T09:00:00" + offset per session.
annotation.valid = True, annotation.warmup = False, annotation.speaker = 1.

── VERIFIKATION ─────────────────────────────────────────────────────────────────

Efter generering: kør

  python3 sources/roc_analysis.py \
    --sweep example_data/synthetic_judged.jsonl \
    --judged example_data/synthetic_judged.jsonl \
    --output figures/

og bekræft at alle fire figurer gemmes uden fejl og at
Cronbach's α printes til stderr.

── KRAV ─────────────────────────────────────────────────────────────────────────

- numpy.random.seed(42) øverst i scriptet — output skal være reproducerbart
- Ingen externe API-kald
- Scriptet skal køre med python3 på macOS med numpy og scipy installeret
```
