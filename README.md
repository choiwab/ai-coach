# AI Sports Coach

An LLM-powered chatbot that bridges **natural language** with **formal verification**. Users ask questions about player/team matchups in plain English; the system extracts statistical parameters from historical data, generates a formal probabilistic model ([PCSP#](https://pat.comp.nus.edu.sg/)), and returns mathematically precise win probabilities — not guesswork.

Built for **CS4211 Formal Methods for Software Engineering** at NUS.

## How It Works

The core idea: instead of asking an LLM to *guess* match outcomes, we use it as an **orchestrator** that calls a formally verified pipeline under the hood.

```
  "Who wins, Djokovic vs Medvedev?"
                │
                ▼
        ┌───────────────┐
        │   Gemini LLM  │  ← decides which tools to call
        │  (coach.py)   │
        └───┬───┬───┬───┘
            │   │   │
   ┌────────┘   │   └────────┐
   ▼            ▼            ▼
analyze_    compare_    get_win_
matchup     parameters  probability
   │            │            │
   ▼            │            ▼
preprocess.py   │       pat_runner.py
   │            │       (manual PAT)
   ▼            │
pcsp_generator.py
   │
   ▼
output/*.pcsp ──────► PAT Model Checker ──► Win Probability
```

### The 3-Stage Pipeline

**Stage 1 — Data Preprocessing** (`preprocess.py`)

Historical shot-by-shot (or play-by-play) CSV data is filtered by player/team and a configurable lookback window (default: 2 years before the match date). From the filtered data, the system counts outcome frequencies — e.g., how many times a player hit a forehand winner from the deuce court. These raw counts become the parameters for the formal model.

**Stage 2 — PCSP Model Generation** (`pcsp_generator.py`)

The extracted frequency counts are injected as `#define` directives into a PCSP# template file. The output is a complete `.pcsp` model file that encodes the probabilistic game dynamics (serves, returns, rallies, scoring) as communicating sequential processes. PAT normalizes the raw counts automatically via `pcase` statements — no manual probability calculation needed.

**Stage 3 — Formal Verification** (`pat_runner.py`)

The generated `.pcsp` file is loaded into [PAT (Process Analysis Toolkit)](https://pat.comp.nus.edu.sg/), which performs probabilistic model checking to compute the exact win probability via reachability analysis. PAT is currently a GUI desktop application, so this step is manual — the system provides step-by-step instructions, and the user feeds the result back.

### LLM Orchestration (`coach.py`)

The Gemini LLM sits on top of the pipeline and uses **function calling** (tool use) to decide when and how to invoke each stage. It has access to three tools:

| Tool | Purpose |
|------|---------|
| `analyze_matchup` | Runs preprocessing + PCSP generation for a given matchup and date |
| `compare_parameters` | Breaks down extracted stats by category (serve, return, rally, etc.) |
| `get_win_probability` | Records the PAT result or provides instructions to run PAT |

The LLM synthesizes tool outputs into natural-language coaching advice, explaining strengths, weaknesses, and tactical recommendations backed by the numbers.

## Setup

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- [PAT](https://pat.comp.nus.edu.sg/) installed (for formal verification step)

### Installation

```bash
git clone <repo-url>
cd ai_coach
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-api-key-here
```

Or export it directly:

```bash
export GEMINI_API_KEY=your-api-key-here
```

## Usage

### Interactive Chat Mode

```bash
python main.py --config configs/example_sport.json
```

This starts a conversational loop where you can ask questions like:
- "Analyze Djokovic vs Medvedev on 2021-02-21"
- "Compare their serve and return stats"
- "What are Djokovic's strengths in this matchup?"

Type `reset` to clear the conversation, `quit` to exit.

### Single Query Mode

```bash
python main.py --config configs/example_sport.json --query "Who wins, Djokovic vs Medvedev?"
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Path to sport config JSON file |
| `--query` | — | Single query (skips interactive mode) |
| `--model` | `gemini-2.0-flash` | Gemini model to use |
| `--output-dir` | `./output` | Directory for generated `.pcsp` files |

## Adding a New Sport

The system is **sport-agnostic**. All sport-specific logic is driven by a JSON config file — no code changes needed.

1. **Prepare your data** as a CSV file (one row per action/shot/play, no header row).

2. **Create a config JSON** specifying:
   - CSV column names and entity columns (player/team identifiers)
   - Date column and lookback window
   - Parameter groups: pandas query strings that define what to count from the data
   - PCSP template file paths

3. **Create PCSP templates**:
   - A variable declarations file (state variables, enums, score counters)
   - One or more process definition files (game dynamics as probabilistic transitions)

4. **Run** with your config:
   ```bash
   python main.py --config configs/your_sport.json
   ```

See [configs/example_sport.json](configs/example_sport.json) and [templates/](templates/) for reference.

### Variants

Some sports require multiple model variants — e.g., tennis has different court geometry depending on player handedness (RH vs LH). Each variant has its own template file and parameter groups. The config supports an `applies_when` field for automatic variant selection based on entity attributes.

## Project Structure

```
ai_coach/
├── main.py              # CLI entry point
├── coach.py             # Gemini LLM wrapper with function-calling loop
├── config.py            # SportConfig / VariantConfig / ParameterGroup dataclasses
├── preprocess.py        # CSV loading, filtering, frequency count extraction
├── pcsp_generator.py    # Assembles .pcsp files from templates + parameters
├── pat_runner.py        # PAT interface (manual mode with instructions)
├── configs/
│   └── example_sport.json
├── templates/
│   ├── example_var.txt      # Variable declarations template
│   └── example_model.txt   # Process definitions template
├── requirements.txt
└── .env                 # GEMINI_API_KEY (not committed)
```

## Key Technical Details

- **Parameters are raw counts, not probabilities.** The extracted values (`p0`, `p1`, ..., `pN`) are integer frequency counts. PAT's `pcase` construct normalizes them into proper probability distributions automatically.
- **Parameter ordering matters.** Entity 1's parameter groups come first, then entity 2's. The split point is `len(params) // 2`.
- **CSV data is cached.** `DataLoader` reads the CSV once and caches the DataFrame in memory to avoid reloading large files (e.g., 1.7GB tennis dataset) on every query.
- **Extracted parameters are cached.** `AICoach._param_cache` stores results keyed by `"{entity1}_vs_{entity2}_{date}"` so tools like `compare_parameters` can access them without re-running the full pipeline.

## Dependencies

See [requirements.txt](requirements.txt). Core dependencies:

- `google-genai` — Google Gemini API client
- `pandas` — Data loading and filtering
- `python-dotenv` — `.env` file support
- `python-dateutil` — Date arithmetic for lookback windows
