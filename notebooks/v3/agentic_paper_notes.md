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

### 4d. First test with a hosted model and real data

- Setting: Colab, VCR rasters from 1985 to 2021, 37 time points, five
  questions in one chat session, 2 November 2026.
- Model availability: `gemini-2.5-flash` returned 404 because Google limited
  it to accounts that had used it before. `gemini-3.8-flash`, the model
  recommended for new projects, returned 503 for high demand on every
  question. In the first question the tool ran in 14.5 s and the model then
  failed after 268.7 s in total, so about 254 s were spent waiting on the
  model, probably in retries by the client library. `gemini-3.6-flash`
  worked.
- Results with `gemini-3.6-flash`:
  1. Class sizes: correct tools; the totals of 1985 and 2021 are both
     4,232,732 pixels, as the common valid mask requires, and the four net
     changes match the data. 21.7 s.
  2. Overall trajectories in pixels: correct tool and argument; the four
     percentages match the totals. 9.4 s.
  3. Overall number of changes from 2007 to 2011: correct custom extent
     tool, plus an unneeded call that returned the full series, which the
     answer did not use. 22.7 s.
  4. Heatmap of a matrix that does not exist: the agent explained the error
     and listed valid keys, but added `interval_1990-2000`, which does not
     exist in this annual series and comes from an example in the
     docstring. The tool returned the error as plain text, so the log
     recorded the call as successful.
  5. Question outside QUEST: refused. 1.9 s.
- Tool selection was correct in 5 of 5 questions, with one unneeded call and
  one invented example.
- Cost grows with the chat: input tokens of the last model call went from
  3,776 in the first question to 15,989 in the refusal, because the chat
  sends the whole history. Evaluation questions must each run in a new chat.
- Log fix: the first version counted model calls over the whole chat
  history, giving 3, 6, 9 and 11. The count now starts after the current
  question.
- Paper relevance: reproducibility of an agent that depends on a hosted
  model is limited by model availability and demand, which changed twice in
  one afternoon. The model must be a parameter, and the evaluation must
  report the model name and date. The test also motivates items 7 to 10:
  tools that return structured errors, a metadata tool with the real keys,
  and docstrings without fixed examples.

### 4e. One result format for every tool

- Problem: tools returned pandas dictionaries with numeric keys turned into
  text, a whole table as a CSV string, or, for charts, only a text that
  described the chart. In the first Colab test, the chart for 2007 to 2011
  gave the model no numbers, so it called a second tool and got the numbers
  of the full series instead of the custom extent.
- Decision: every data tool and every chart tool returns one dictionary
  with the unit, the time extent, the number of time points, the study area,
  readable column names and the rows. Chart tools add the numbers shown in
  the chart. Matrices carry class names instead of class codes, and
  trajectories carry the labels of the trajectory map legend. The data tools
  include trajectory 1, so trajectories 1 to 5 add up to the study area.
  The conversion is done by helper functions in the chat cell; the QUEST
  modules and `quest_core.py` did not change.
- Evidence: 22 of 22 checks passed on 10 synthetic rasters with 8 classes.
  The rows of every tool match the CSV tables written by the engine; class
  sizes add up to the study area in every year; a custom extent from 2003 to
  2006 reports its own time extent and never more than 3 changes per pixel;
  filters by year and interval return only the requested rows.
- Token cost: one dictionary per row made the change components table four
  times the size of the former CSV string, 47,086 characters. Sending a
  header and lists of values cut it to 18,112 characters. Because the chat
  resends its whole history, the size of tool results drives the cost of
  every later question.
- Paper relevance: the format of tool results is a design decision with
  measurable effects on correctness, which numbers the model uses, and on
  cost, which tokens the model reads.

### 4f. A metadata tool so the agent does not guess keys

- Problem: in the first Colab test the agent listed `interval_1990-2000` as
  a valid matrix key. It does not exist in the annual series; the model took
  it from an example in the tool docstring. The agent had no tool that
  described the dataset itself.
- Decision: a 21st tool, `get_dataset_metadata`, returns the years, the
  intervals, the classes with codes and names, the NoData value, the valid
  matrix keys, the trajectory labels, the raster grid and the warnings of
  the input check. It reads only the configuration and does not run the
  analysis. The fixed interval example was removed from the docstrings of
  the matrix tools, which now point to the metadata tool, and the system
  instruction tells the agent to use it and never guess years, intervals,
  classes or keys.
- Evidence: on synthetic data the matrix keys built by the tool equal the
  keys produced by the engine, in the same order; the result has about
  1,400 characters; the QUEST cache stays empty after the call. The 22
  checks of the result format still pass. Not yet tested with the model.
- Paper relevance: grounding the agent in a description of the data, rather
  than in examples written in the code, is a direct answer to invented
  values. The test question with a nonexistent matrix key measures it.

### 4g. Second Colab test, after the result format and the metadata tool

- Setting: Colab, VCR rasters from 1985 to 2021, the same five questions,
  3 November 2026, spread over several chat sessions because of service
  errors. Questions 1 to 3 and 5 ran with `gemini-3.6-flash`; question 4
  and a second refusal ran with `gemini-3.5-flash-lite`.
- Service errors in one day: five 402 errors after the prepaid credits of
  the first account ran out, on both `gemini-3.6-flash` and
  `gemini-3.8-flash`; one 503 for high demand on `gemini-3.6-flash` after
  23.3 s; one 404 on `gemini-2.5-flash-lite`, no longer open to new users.
  A second account on the free tier worked, with a limit of 5 requests per
  minute for `gemini-3.6-flash`. One question with tools used 4 of those 5
  requests and about 21,000 tokens in its minute, according to the AI
  Studio usage page, while the log recorded 10,705 input tokens for the
  last model call only.
- Results:
  1. Class sizes: metadata, class sizes and the class size chart. Values
     and shares match the data, for example Water at 3,626,433 pixels and
     85.68% in 1985. 70.3 s in total, of which 10.7 s ran the QUEST engine.
  2. Overall trajectories in pixels: chart plus the trajectory data tool.
     The five trajectories add up to the study area of 4,232,732 pixels;
     trajectory 1 has 4,039,089 pixels, 95.43%. 10.6 s.
  3. Overall number of changes from 2007 to 2011: metadata and the custom
     extent chart only. The answer gives the numbers of the custom extent,
     from 28,173 pixels with one change to 17 pixels with four changes,
     where the first test gave no numbers and made an unneeded call for the
     full series. 7.2 s.
  4. Heatmap of a matrix that does not exist: the agent called the metadata
     tool, did not call the heatmap tool, and listed only keys that exist,
     such as `interval_1985-1986`. The invented key of the first test did
     not appear. 2.0 s.
  5. Question outside QUEST: refused by both models.
- Remaining issues: in questions 1 and 2 the agent still called a data tool
  in addition to a chart tool that already returned the same numbers, which
  doubles the tokens of the tool results. Calling the metadata tool first
  adds one model call per question, which on a 5 requests per minute limit
  is a real cost.
- Paper relevance: the result format fixed the custom extent answer and the
  metadata tool removed the invented key, measured on the same questions
  before and after. Grounding has a cost in model calls, and the free tier
  limits the agent to about one question per minute, a constraint set by
  the number of model calls per question, not by the QUEST computation.

### 4h. Errors the agent and the user can act on

- Problem: three kinds of errors reached the user or the model in a form
  nobody could act on. Service errors of the Gemini API appeared as raw
  JSON; the two tests produced 402, 404 and 503 errors in two days. A tool
  that raised an exception was converted by the client library into a long
  technical message. A tool that returned an error as a result was recorded
  as a success, as in the nonexistent matrix of the first test. Some invalid
  arguments failed silently: a chart metric other than percentage or pixels
  was drawn in pixels without a warning.
- Decision: the tool decorator returns a short error instead of raising,
  and records any result with an error as a failure. Seven tools with
  parameters check their arguments before running: the chart metric, the
  order of the years, both years given or neither, at least two years of the
  dataset inside a custom extent, and requested years present in the data.
  Each message states the valid options, such as the years of the dataset.
  Service errors are translated into a short message with the action to take,
  for 402, 404, 429, 5xx, a rejected key and a failed connection; for 404 the
  suggested model is read from the API message. The log keeps the code, the
  full error and the message shown. The system instruction asks the agent to
  explain tool errors with the options they give and never report numbers
  no tool returned.
- Evidence: 13 of 13 checks passed on synthetic data, including the short
  error reaching the model through the function-calling code of the client
  library, and the 22 checks of the result format still pass. Not yet tested
  with the model.
- Paper relevance: in a hosted agent, failures of the service are as frequent
  as failures of reasoning. Turning both into messages with an action is part
  of the design, and the log makes their frequency measurable.

### 4i. Docstrings and a system instruction with the QUEST definitions

- Problem: the docstrings said little more than the name of each tool, so
  the model could not tell when to use a tool, what unit it returns, or that
  two similar tools differ: the interval chart filtered by years counts the
  changes of the whole series, while the custom extent tool recomputes the
  analysis. The system instruction defined the scope but not the method, and
  the second Colab test showed the agent calling a data tool after a chart
  tool that had already returned the same numbers.
- Decision: the docstrings of the 20 analysis tools were rewritten to say
  what each tool returns, in which unit, when to use it, and the valid values
  of each parameter; chart tools state that they return the numbers they
  show. The system instruction is now a constant, `SYSTEM_INSTRUCTION`, with
  rules of work and the QUEST definitions: time interval, number of changes,
  extent, alternation exchange and shift, indirect extent, the components of
  the extent and of each interval, and the five trajectories. The
  definitions were written from the engine code and reviewed by the author.
  Answers use "locations" instead of "pixels"; columns and options keep the
  name "pixels" as identifiers.
- Evidence: the schemas of the 21 tools still build; descriptions add up to
  5,900 characters and the instruction to 2,950. The earlier checks still
  pass: 22 for the result format, 13 for errors, and the matrix keys equal to
  those of the engine. Not yet tested with the model.
- Paper relevance: the instruction carries the domain knowledge the agent
  needs, written once and reviewed by the method author, while every number
  still comes from the engine. Keeping it in one constant lets the
  evaluation compare versions of the instruction.

### 4j. Third Colab test, after errors, docstrings and definitions

- Setting: Colab, VCR rasters from 1985 to 2021, seven questions in one
  chat session with `gemini-3.5-flash-lite`, 4 November 2026. An earlier
  attempt with `gemini-3.6-flash` failed with 503 after 31.5 s; the chat
  showed the short message written for that code, and the log kept the code,
  the full error and the two tool calls made before the failure.
- Results:
  1. Class sizes: metadata and the class size chart only, with no repeated
     data tool. Values match the data. 7.8 s.
  2. Overall trajectories in locations: the chart tool only. Trajectories 2
     to 5 are correct. The agent added trajectory 1 by subtracting from the
     study area, 4,039,089 locations, which is right, but gave 95.38% where
     the data give 95.43%: a share computed by the model, against the rule of
     the system instruction. 2.1 s.
  3. Overall number of changes from 2007 to 2011: the custom extent tool
     only, with the same numbers as the second test. 4.5 s.
  4. Heatmap of a matrix that does not exist: metadata only; valid keys
     listed, including `interval_1985-1986`. 2.2 s.
  5. Question outside QUEST: refused. 0.9 s.
  6. Chart in hectares: the tool returned the metric error, the log marked
     it as a failure, and the agent gave the two valid options. 1.4 s.
  7. Number of changes from 1970 to 1980: the tool returned the year range
     error, marked as a failure, and the agent gave the years of the
     dataset. 38.0 s, almost all of it waiting for the model.
- Every answer used "locations". The chat input tokens of the last model
  call grew from 7,839 to 10,825 over the seven questions.
- Remaining issue: the trajectory chart returns trajectories 2 to 5, so a
  question about trajectory 1 leads the model to compute its share. The
  chart tool can return trajectory 1 as well, marked as not plotted.
- Paper relevance: before and after on the same questions, the repeated data
  call of questions 1 and 2 disappeared, and the two new error cases were
  explained with valid options. The one wrong number came from arithmetic
  done by the model, which supports keeping every statistic in the tools.

### 4k. Areas in square kilometers and hectares, and trajectory 1 in charts

- Problem: results were in locations and percent only, so a question about
  area required the model to multiply by a pixel area it did not have. In
  the third test the model also computed the share of trajectory 1 itself,
  because the trajectory chart returned trajectories 2 to 5 only, and gave
  95.38% instead of 95.43%.
- Decision: the area of one pixel is computed once from the first raster.
  In a projected coordinate reference system it is the spatial resolution
  along x times along y, converted to square meters, and it is exact. In a
  geographic coordinate reference system it is the geodesic area of the
  pixel at the center of the map; when the pixel area changes by more than
  1% between the top and bottom rows, every result carries a note that areas
  are approximate. Every count of locations in every result gets two
  columns, in square kilometers and hectares, and each result gives the
  study area in both units. Charts keep their axes in locations or percent;
  areas on chart axes are left for a later version. The trajectory chart
  tools now return trajectories 1 to 5, with a column that marks trajectory
  1 as not plotted.
- Evidence: 14 of 14 checks passed on synthetic data. A 30 m grid in UTM
  gives 900 m2 per pixel, so 284,000 locations give 255.6 km2 and 25,560 ha;
  a 0.00025 degree pixel at the equator gives 769.3 m2; a geographic raster
  from 0 to 20 degrees of latitude is flagged as approximate, with 1.5%
  variation. The earlier checks still pass.
- Cost: the area columns raise the size of the change components table from
  18,112 to 26,038 characters. Both units were kept so the model does not
  convert between them.
- Paper relevance: unit conversion is moved out of the model, following the
  third test, where the one wrong number came from arithmetic by the model.
  The treatment of geographic grids is a stated approximation, reported to
  the user rather than hidden.

### 4l. A client-side limit on model requests per minute

- Problem: the free tier allows 5 model requests per minute for the models
  tested, and one question makes 1 to 4 model calls, so two questions within
  a minute can exceed the limit and return 429.
- Decision: before each question, the chat counts the model calls of the
  last minute and, when those calls plus 3 expected for the new question
  would pass `REQUESTS_PER_MINUTE`, waits only until enough calls are older
  than a minute and says so in the chat. The wait is logged apart from the
  response time. The limit is a parameter of cell 2.1; 0 turns it off.
- Evidence: with a simulated clock, a question 10 s after one with 3 calls
  waits 51 s; a question after the window has cleared does not wait; a limit
  of 0 never waits. In the notebook harness the wait was 61 s, as computed.
- Limitation: the expected number of calls is an estimate, so a question
  with more calls can still receive 429, which the error message of item 9
  explains.
- Paper relevance: the limit is set by the number of model calls per
  question, not by the number of questions, which ties the throughput of the
  agent to its tool-calling behavior.

### 4m. One tool for questions about one class

- Problem: a question about one class, such as how Marsh changed, needed
  three or four tool calls: class sizes, change components, the extent
  matrix and the time intervals matrix. Each call costs a model request,
  and on the free tier the limit is 5 per minute.
- Decision: `get_class_summary` returns, in one result of about 2,100
  characters, the size of the class at the first and last time points, the
  net change, the gain and loss summed over every time interval, the gain
  and loss in each of the five components of the whole time extent, and the
  transitions from and to the other classes between the first and last time
  points, all with areas. The class name is matched without regard to case,
  and an unknown name returns the list of classes. The system instruction
  points questions about one class to this tool.
- Evidence: 11 of 11 checks passed on synthetic data, against the CSV tables
  of the engine: sizes, interval gain and loss, the five components, and the
  extent transitions; the first size plus gains minus losses equals the last
  size, and the Quantity gain minus loss equals the net change. The earlier
  checks still pass.
- Paper relevance: tools shaped around the questions users ask reduce the
  number of model calls per question, which sets both cost and throughput.

### 4n. A wording in a tool result led to a wrong interpretation

- Test: "How did Marsh change? Give the components and where it went", with
  `gemini-3.5-flash-lite`, 4 November 2026. The agent made one analysis
  call, `get_class_summary`, and every number was consistent: 489,708
  locations in 1985 minus 54,672 lost in the extent plus 31,125 gained gives
  466,161 in 2021; the Quantity loss equals the net change of 23,547; the
  gain summed over the intervals, 94,688, minus the extent gain gives the
  alternation of 63,563, equal to Alternation Exchange plus Alternation
  Shift. 27.2 s.
- Error: the agent said that 435,036 locations "remained Marsh throughout
  the entire time extent". That value is the diagonal of the extent matrix,
  the locations that were Marsh at the first and last time points, which
  includes locations that changed and came back. The note of the tool called
  it "persistence", and the model read persistence as no change at every
  time point.
- Fix: the note now says that this transition counts locations in the class
  at both the first and last time points, including those that changed and
  came back, so it is not persistence through every time point.
- Paper relevance: correct numbers can still be misread when the label a
  tool attaches to them is ambiguous. Every term in a tool result is part of
  the interface and needs the same precision as the method definitions.

### 4o. Maps of Extent, Alternation Exchange and Alternation Shift

- Problem: the engine already wrote the three component rasters, but the
  agent could only show the maps of number of changes and trajectories.
- Decision: the maps module gained a component map with the titles and
  legend titles of the 03_quest_pie notebook, drawn like the number of
  changes map, and a block-by-block count of the locations of each map
  value, so large rasters are never read whole. A 23rd tool,
  `plot_change_component_map`, shows the map of extent, alternation exchange
  or alternation shift and returns the locations of each value, with areas.
- Evidence: 10 of 10 checks passed on synthetic data. The counts of each map
  add up to the study area; the locations with extent change equal the
  off-diagonal total of the extent matrix, 170,140; the sum of value times
  locations equals the Alternation Exchange of the overall components,
  24,610, and the Alternation Shift, 74,115, which confirms that the map
  already leaves out the indirect extent, as the components table does.
- Paper relevance: the maps and the tables come from the same engine run,
  and the tests tie each map to the matching table, so a map shown by the
  agent agrees with the numbers it reports.

### 4p. Fourth Colab test, on another VCR dataset

- Setting: Colab, `gemini-3.5-flash-lite`, six questions, 8 November 2026.
  By mistake the test used another VCR dataset, with a study area of
  3,878,017 locations, so its numbers cannot be compared with the earlier
  tests. The behavior of the agent does not depend on the data, and every
  answer was checked against the tool results of the same session.
- Results:
  1. Area of Marsh in hectares: metadata and `get_class_summary`, with the
     note that areas are approximate. 17.5 s, of which 15.1 s ran the engine.
  2. Share of locations with no change: read from the trajectory chart
     tool, which now returns trajectory 1, as 97.963%, with no arithmetic by
     the model. 2.0 s.
  3. How Marsh changed: one call to `get_class_summary`; every number is
     consistent, such as 319 minus 10 plus 2 giving 311. The agent no longer
     said that the persisting locations stayed Marsh at every time point,
     but still used the word "persisting". 2.3 s.
  4. Extent map: one call. The request limit waited 8.0 s before the
     question. 3.6 s.
  5. Alternation Exchange map and the number of locations with it: the
     agent added the rows itself; the sum, 22,716 locations, and the share,
     0.5858%, were right, but the area, 16.19 ha, was wrong by a factor of
     100. It also made an unneeded call to `get_overall_number_of_changes`.
  6. Map of Quantity: the agent did not call the map tool, which would have
     returned the error, and made an unneeded call to `get_class_summary`
     for Trees, but it explained that Quantity has no map and gave its value
     from the components tool. The request limit waited 29.2 s.
- Fix: the map tool now returns two totals, with areas and shares: the
  locations with a value above 0, and the changes in the component, the sum
  of value times locations. On synthetic data both totals match the sums of
  the rows, and the earlier 10 checks still pass.
- Paper relevance: the second wrong number in four tests came, like the
  first, from arithmetic done by the model on rows the tool returned. Each
  time the fix was to return the needed total from the tool.
- Retest of question 5 on the VCR dataset of the earlier tests, the same
  day: one analysis call; the answer took the total from the tool, 68,491
  locations with alternation exchange, 1.6181% of the study area of
  4,232,732 locations and 4,882.88 ha, which matches the rows of the same
  result. 14.3 s, of which 11.7 s ran the engine.

### 4q. Custom time extents for matrices, components and class summaries

- Problem: the custom time extent existed only for the charts of number of
  changes and trajectories. A question such as the components of change
  from 2003 to 2006, or the transitions of one class in that period, could
  not be answered with the numbers of that period.
- Decision: seven tools gained optional start and end years: the two
  transition matrix tools, the four change component tools and the class
  summary. With both years, the tool builds a new module of the same type
  restricted to those years, so the engine runs again with the common valid
  mask of that period, and the result states the period used. Without years
  nothing changes. The year checks of item 9 apply, and an interval matrix
  key outside the period returns an error that says so.
- Evidence: 13 of 13 checks passed on synthetic data, against an
  independent computation with numpy on the rasters for 2003 to 2006 with
  the same common mask: the extent matrix, an interval matrix, the study
  area and the first and last sizes and transitions of one class are equal;
  Quantity plus Allocation Exchange plus Allocation Shift equals the extent
  change of 79,243 locations. All earlier checks still pass.
- Paper relevance: the time extent is a parameter of the analysis that the
  user sets in words, and the agent passes it to the engine instead of
  filtering numbers of the whole series, which would give wrong components
  for the period.

### 4r. Fifth Colab test, custom time extents

- Setting: Colab, the VCR dataset of the earlier tests, `gemini-3.5-flash-lite`,
  three questions, 9 November 2026.
- Results:
  1. Overall change components from 2007 to 2011: one analysis call with
     the period. The five components add up to 35,264, which equals the
     total number of changes in the custom extent chart of the second test,
     28,173 + 2 x 3,078 + 3 x 289 + 4 x 17, a check across two tools and two
     sessions. 10.7 s.
  2. How Marsh changed between 2000 and 2010: one call,
     `get_class_summary("Marsh", 2000, 2010)`. Every number is consistent:
     484,594 minus 20,174 lost plus 10,703 gained gives 475,123; the
     Quantity loss equals the net change of 9,471; the interval gains minus
     the extent gains, 10,649, equal Alternation Exchange plus Alternation
     Shift. The agent described the diagonal correctly, as locations in
     Marsh at both time points including those that changed and returned,
     after the fix of entry 4n. One heading was mislabeled, "Other classes
     gained from Marsh" for gains of Marsh. The request limit waited 44.1 s.
  3. Heatmap of an interval outside the period: the tool returned the
     error, the agent listed the four intervals of 2007 to 2011 and offered
     the full series instead. 3.1 s.
- Paper relevance: numbers from different tools and sessions agree when
  they describe the same quantity, which the evaluation can use as an
  internal consistency check that needs no reference answer.

### 4s. Saved charts and explicit retries on server errors

- Review of the robustness items: the engine already shows a progress bar
  for every run, including custom time extents, so nothing changed there.
  Maps already saved their files, but the nine chart methods of the class
  size, number of changes, trajectory, transition and component modules only
  showed the figure, which was lost when the session ended. Without an
  explicit setting, the client library made no or implicit retries on
  server errors, which matches the uneven waits seen in the tests: 254 s on
  one day, 23 s and 31 s on others.
- Decision: while a chart tool runs, the figure is saved to
  OUTPUT_PATH/charts before it is shown, and the result gives the file in
  "chart_file"; the chart modules did not change. The client now retries a
  failed model call up to 3 times on 500, 503 and 504, waiting about 5 s and
  then about 10 s, with a maximum of 30 s; 429 is left to the request limit
  of entry 4l. A retry repeats only the model call that failed, not the
  tools already run.
- Evidence: all 11 chart tools wrote a PNG and returned its path; error
  results and map tools got no chart file; matplotlib was restored after
  each call. A local server that returned 503 twice and then answered gave
  the answer after requests at 0.1, 5.4 and 15.5 s; three 503 gave the error
  after 15.9 s; a 429 was not retried. All earlier checks still pass.
- Paper relevance: bounded, stated retries make latency under service errors
  predictable and reportable, and saved charts make every answer of an
  evaluation run auditable after the session.

### 4t. Evaluation design and scoring

- Design agreed for Phase 5: 60 questions in eight categories, one number,
  custom time extent, one class, charts and maps, invalid input, outside
  QUEST, comparison, and six translations into Portuguese and Spanish of
  questions in the other categories. Questions are templates filled from
  the metadata of each dataset; the four classes are the four largest by
  number of locations at the first time point, so the same set runs on any
  legend. Datasets: VCR and PIE, with western Bahia as an option. Models:
  `gemini-3.5-flash-lite` for the full evaluation, 60 questions with 2
  repetitions per dataset, about 720 requests within its free limit of 500
  per day and 15 per minute; `gemini-3.6-flash`, with a free limit of 20 per
  day, on a subset of 15 questions with 1 repetition. Each question runs in
  a new chat, so the history of earlier questions does not change the
  answer or the cost.
- References: for each question, the expected tools and arguments, and the
  numbers that must appear in the answer, computed with the same functions
  the tools use, without the model. The engine was validated separately
  against the 03_quest_pie notebook and numpy, so the evaluation measures
  the agent, not QUEST.
- Scores per answer: tool selection, with every expected tool called with
  the expected arguments; extra calls; share of reference numbers found in
  the answer; numbers of the answer with no source in the tool results of
  the same question; behavior, an answer, an explained error or a refusal;
  language of the answer; model calls, tokens and time; and service errors,
  kept apart. A number counts as grounded when it equals a number of the
  results, or the sum or difference of two numbers of the same column, or
  the sum of three numbers of a small column. Numbers are read with either a
  point or a comma as the decimal separator, and a separator counts as
  thousands only before groups of exactly three digits.
- Evidence: with a scripted fake model on synthetic data, every injected
  fault was detected: a value in square kilometers given as hectares, a
  wrong tool, a wrong period, an answer in the wrong language, a tool call on
  a question outside QUEST, an extra call, and a 503 that was asked again on
  the next run. Correct sums and differences computed by the model were not
  flagged.
- Limitation: a number in the wrong unit can still match a number of the
  results, so grounding alone does not catch unit errors; the reference
  check does.
- Transparency: the evaluation lives in Section 6 of the notebook, with the
  questions, references and scoring visible, and runs only when switched on.

### 4u. First evaluation run on VCR, stopped by the per-minute limit

- Setting: Section 6 on Colab, VCR, `gemini-3.5-flash-lite`, 9 November 2026.
  The run answered 19 questions, A1 to B7, in 3.5 minutes and stopped at B8
  with a 429 for the limit of 15 requests per minute, although the request
  limit of the chat had counted at most 15 calls in the last minute. The
  runner stopped as designed for a daily limit, which this was not.
- Scores of the 19 answers: every expected tool called with the expected
  arguments except one, every reference number found, and one answer with a
  number that no tool returned. Median time 3.1 to 3.8 s per question.
- Review of the flagged answers:
  1. A10, Alternation Exchange over the whole series: the agent used the
     component map, whose totals give the same 167,896 changes as the
     components chart. A valid choice, so the map tool was added to the
     accepted tools of A10.
  2. A12, locations that were Water in 1985 and 2021: the number was right,
     3,587,126, and the agent described it correctly as including locations
     that changed and came back, but it computed the share, 84.75%, itself,
     because matrix rows had no percent column. The share is correct, but
     it breaks the rule that every statistic comes from a tool.
- Fixes: matrix rows and the transitions of the class summary now include
  the percent of the study area. The request limit of the chat now expects
  4 model calls per question instead of 3. The runner reads the error of a
  429: for the per-minute limit it waits the delay given by the service and
  asks again, up to 3 times; it stops only for a daily limit or missing
  credits.
- Paper relevance: automatic flags needed a human reading to tell a valid
  alternative tool from a wrong one, and a correct share computed by the
  model from a missing column; both led to changes in the tools or the
  reference, not in the scores by hand.

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
