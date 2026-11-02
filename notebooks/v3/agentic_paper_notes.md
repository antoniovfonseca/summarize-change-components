# Agentic QUEST paper notes

Working notes for a manuscript on the agentic version of QUEST, a candidate
for the Transactions in GIS special issue "Perspectives and Advances in
Autonomous GIS", with a submission deadline of 31 October 2027. The paper
focuses on the agent architecture and its evaluation. The QUEST method is
described in a separate manuscript and cited here.

Each entry records a design decision, the problem it addresses, and the
evidence gathered so far. Entries that are not yet tested say so.

## Architecture

- The language model selects and calls tools. It never writes code and never
  computes a number.
- Every number comes from `quest_core.py`, a deterministic engine shared with
  the conventional notebook `03_quest_pie.ipynb`.
- The engine reads each raster block once and computes trajectories, numbers
  of changes, interval transition matrices, change component matrices, class
  counts, and the extent, alternation exchange and alternation shift rasters.
- The agent runs the engine once per set of years and caches the results, so
  later questions read from the cache.
- The user supplies the rasters. The notebook no longer depends on a private
  cloud bucket.

## Decision log

### 1. One computational engine for the agent and the conventional notebook

- Problem: before the integration, the agentic notebook had its own
  implementation of the QUEST computation, and it had diverged from the
  reference notebook without any warning. On synthetic data, its overall
  Alternation Shift was 113253 transitions against 65658 in the reference,
  72% higher, because it did not subtract the Indirect Extent. Its mask also
  removed only NoData, so a value outside the legend entered the matrices as
  an extra class, and the class totals differed between years.
- Decision: move the computation of the reference notebook into
  `quest_core.py` and make the agent call it.
- Evidence: on synthetic rasters with NoData in one year and an unlisted class
  in another year, the 27 tables written by the agent are byte-identical to
  those of the reference notebook. Class sizes, numbers of changes,
  trajectories and the change component summary match.
- Paper relevance: an agent that reimplements an analysis can drift from the
  method silently. Sharing one engine gives numeric fidelity by construction,
  so the evaluation can focus on tool selection and on how the agent reports
  the numbers.

### 2. Cross-implementation testing found errors in the reference as well

- Problem: the comparison between implementations also exposed errors in the
  reference notebook. After one commit, the alternation exchange raster was
  written as NoData everywhere. The scale bar of the output maps ignored the
  display downsampling, so it was wrong by the downsampling factor, for
  example 50% off with a factor of 2 and 83% off with a factor of 6, and its
  label wrote "km" for bars measured in meters.
- Decision: fix the reference, then rerun the comparison.
- Evidence: after the fixes, the scale bar matches the geodesic reference in
  UTM, geographic and Web Mercator test rasters, and the exchange raster
  matches the change component matrices.
- Paper relevance: the evaluation of an autonomous system depends on a
  reference that is itself tested. Running two implementations against the
  same data is a practical way to test both.

### 3. Common valid mask for every product

- Problem: different outputs used different sets of locations. Some tables
  counted intervals of locations with NoData in some years, so a percentage
  could have a numerator and a denominator from different sets.
- Decision: a location enters every output only if it is valid and belongs to
  a listed class in every year. The agent reports this as the study area.
- Evidence: on synthetic data, the matrices, rasters, trajectory counts and
  class sizes all describe the same number of locations.
- Paper relevance: a single, stated definition of the study area lets the
  agent answer questions about area without ambiguity.

### 4. User-provided data with checks the agent can explain

- Problem: with any user's data, the agent needs to detect input problems
  before running the analysis.
- Decision: the user sets a folder, a file name pattern of prefix, year and
  suffix, the years, the NoData value and the legend. Cell 2.4 checks the
  files and returns messages written so the agent can relay them, such as a
  year without a file or a prefix that matches no file.
- Evidence: the messages were tested for a missing year, a wrong prefix and
  unsorted years.
- Paper relevance: checking fitness for use is a task of the agent, which
  connects to the call's topics on data provenance and fitness for use.

### 4b. Input checks that block silently wrong results

- Problem: the engine stores class codes and NoData as 8-bit integers. A
  class code or NoData value outside 0 to 255, such as 1000 or -1, would be
  converted without warning and produce wrong numbers instead of an error.
  Rasters with the same size but a different CRS or transform would also be
  compared pixel by pixel without any error, although the pixels do not
  cover the same locations.
- Decision: cell 2.4 checks every raster before any computation. Errors stop
  the run and are reported together in one message: a file that cannot be
  opened; width, height, CRS or transform that differ from the first year; a
  non-integer data type; NoData, class codes or raster values outside 0 to
  255. Warnings let the run continue: more than one band, a NoData value
  declared in the file that differs from the one set by the user, values
  absent from the legend, whose locations leave the common valid mask, and
  legend codes absent from the data. Raster values are listed from a
  decimated sample of at most 2000 pixels on the longer side, so a rare class
  can be missed, and the message says it comes from a sample.
- Evidence: 12 synthetic cases each gave the expected error or warning. The
  check of two 6000 by 5000 rasters took 0.05 s. The report is stored in the
  configuration, so a later metadata tool can give the grid, CRS and warnings
  without reopening the files.
- Paper relevance: an agent that accepts user data must refuse to compute
  when the input would give silently wrong results, and explain why. This is
  a guardrail placed before the language model, not inside it.

### 4c. Tool-call log written by the code, not by the model

- Problem: the chat printed only the final answer, so there was no record of
  which tools the agent called, with which arguments, or whether it called
  any tool at all. Without that record, tool selection cannot be evaluated.
- Decision: every tool is wrapped by a decorator that records its name,
  arguments, run time and status. A function `ask_quest` sends each
  question, times the answer and writes one JSON line per question to
  `OUTPUT_PATH/logs/chat_<date-time>.jsonl`, with the question, the answer
  or the error, the model, the number of model calls, the token counts and
  the tool calls. The chat shows a short line with the tools after each
  answer. The evaluation runner of Phase 5 will call the same `ask_quest`,
  so the chat and the evaluation produce the same record.
- Evidence: the schema that `google-genai` builds from each function is
  identical with and without the decorator for the 20 tools, so the model
  sees the same parameters and docstrings. Calls passed through the library's
  own function-calling code were recorded with their arguments; a year sent
  by the model as 2007.0 is converted to the integer 2007 before the tool
  runs.
- Findings from reading the `google-genai` code: (1) when a tool raises an
  exception, automatic function calling catches it and sends the model a
  long technical message, so the agent already receives errors, but in a
  form it cannot explain well to the user; this is the scope of item 9.
  (2) A question that uses tools makes two or more model calls, and the
  response reports the tokens of the last call only. Exact cost per
  question needs either a manual function-calling loop that sums the
  tokens of every call, or a cost reported as approximate. To be decided in
  Phase 5.
- Paper relevance: the log is written by the code, so the model cannot
  leave out or invent a tool call. It is the basis for the tool-selection,
  latency and cost metrics of the evaluation.

### 5. Bounded autonomy with controlled flexibility

- Problem: an agent limited to fixed tools can look rigid, while an agent that
  writes code cannot guarantee correct numbers.
- Decision: the agent chooses among fixed tools, and some tools accept
  parameters that recompute the analysis, such as a time extent chosen by the
  user.
- Evidence: for a time extent from 2007 to 2011, the common valid mask is
  recomputed with those years only, giving 288000 locations against 284000
  for the full series in the synthetic test.
- Paper relevance: an example of an intermediate level of autonomy, which the
  call asks contributors to discuss.

### 6. Cost and latency

- Problem: the first question triggers the full computation.
- Decision: cache the results per set of years.
- Evidence: not yet measured.
- Paper relevance: response time should be reported separately for the first
  question and for later questions.

## Planned evaluation

- A set of 50 to 100 questions, each with the expected tool and reference
  values generated by the engine.
- Metrics: tool selection, agreement between the numbers in the answer and
  the engine, and statements not supported by the data.
- Repeated runs with at least two language models and at least two datasets,
  such as PIE and VCR.
- A log of the tools called in each answer, needed to measure tool selection.

## Open questions

- Whether to extend the agent to the Google Earth Engine version of QUEST as a
  second engine, which would add data discovery across global products.
- Which public sample dataset to archive for the data availability statement.
