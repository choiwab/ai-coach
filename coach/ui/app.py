from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from coach.agent.llm_client import LLMClient
from coach.agent.planner import AgentExecutor
from coach.service import BadmintonCoachService

load_dotenv(override=True)

app = FastAPI(title="AI Badminton Coach")
service = BadmintonCoachService()
llm_client = LLMClient()
chat_executor = AgentExecutor(service=service, llm_client=llm_client)


class PredictRequest(BaseModel):
    a: str
    b: str
    mode: str = Field(default="mock", pattern="^(mock|real)$")
    window: int = Field(default=30, ge=1, le=500)


class StrategyRequest(BaseModel):
    a: str
    b: str
    mode: str = Field(default="mock", pattern="^(mock|real)$")
    window: int = Field(default=30, ge=1, le=500)
    budget: int = Field(default=60, ge=1, le=1000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    mode: str = Field(default="mock", pattern="^(mock|real)$")
    window: int = Field(default=30, ge=1, le=500)
    budget: int = Field(default=60, ge=1, le=1000)
    show_trace: bool = False


def _render_home() -> str:
    example_players = service.adapter.players_df["name"].head(8).tolist()
    example_json = json.dumps(example_players)
    return f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI Badminton Coach</title>
        <style>
          :root {{
            --bg: #0c1b17;
            --bg-soft: #132822;
            --panel: rgba(15, 35, 30, 0.92);
            --panel-strong: rgba(20, 44, 38, 0.96);
            --ink: #eef7f2;
            --muted: #9bb7ad;
            --line: rgba(213, 243, 229, 0.14);
            --court: #1d6a47;
            --court-deep: #0f4f33;
            --court-line: rgba(233, 247, 240, 0.84);
            --accent: #ffb000;
            --accent-2: #35d08d;
            --accent-3: #77d7ff;
            --danger: #ff7b5c;
            --shadow: 0 28px 70px rgba(0, 0, 0, 0.34);
          }}

          * {{
            box-sizing: border-box;
          }}

          body {{
            margin: 0;
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            color: var(--ink);
            background:
              radial-gradient(circle at top left, rgba(53, 208, 141, 0.12), transparent 22%),
              radial-gradient(circle at top right, rgba(119, 215, 255, 0.1), transparent 24%),
              linear-gradient(180deg, #10231d 0%, var(--bg) 100%);
          }}

          .shell {{
            width: min(1180px, calc(100% - 32px));
            margin: 24px auto 40px;
          }}

          .hero {{
            position: relative;
            overflow: hidden;
            padding: 34px;
            border: 1px solid var(--line);
            border-radius: 28px;
            background:
              linear-gradient(145deg, rgba(14, 47, 38, 0.98), rgba(10, 27, 23, 0.98));
            box-shadow: var(--shadow);
          }}
          
          .hero::after {{
              content: "";
              position: absolute;
              right: 34px;
              top: 26px;
              width: 150px;
              height: 240px;
              border-radius: 0;
              background:
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='240' viewBox='0 0 150 240'%3E%3C!-- Court surface --%3E%3Crect width='150' height='240' fill='%231a5c32' fill-opacity='0.55'/%3E%3C!-- Outer border --%3E%3Crect x='1' y='1' width='148' height='238' fill='none' stroke='rgba(255,255,255,0.85)' stroke-width='2'/%3E%3C!-- Horizontal lines --%3E%3Cline x1='1' y1='13.7' x2='149' y2='13.7' stroke='rgba(255,255,255,0.85)' stroke-width='1.5'/%3E%3Cline x1='1' y1='84' x2='149' y2='84' stroke='rgba(255,255,255,0.85)' stroke-width='1.5'/%3E%3Cline x1='1' y1='120' x2='149' y2='120' stroke='rgba(255,255,255,0.85)' stroke-width='2.5'/%3E%3Cline x1='1' y1='156' x2='149' y2='156' stroke='rgba(255,255,255,0.85)' stroke-width='1.5'/%3E%3Cline x1='1' y1='226.3' x2='149' y2='226.3' stroke='rgba(255,255,255,0.85)' stroke-width='1.5'/%3E%3C!-- Vertical center line --%3E%3Cline x1='75' y1='84' x2='75' y2='156' stroke='rgba(255,255,255,0.85)' stroke-width='1.5'/%3E%3C!-- X data points --%3E%3Ctext x='30' y='50' font-size='14' font-weight='bold' fill='%23ff4d6d' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='110' y='40' font-size='14' font-weight='bold' fill='%2300cfff' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='70' y='68' font-size='12' font-weight='bold' fill='%23ffd60a' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='45' y='105' font-size='13' font-weight='bold' fill='%2300cfff' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='112' y='98' font-size='12' font-weight='bold' fill='%23ff4d6d' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='88' y='140' font-size='14' font-weight='bold' fill='%23ffd60a' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='35' y='172' font-size='12' font-weight='bold' fill='%23a8ff78' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='105' y='165' font-size='13' font-weight='bold' fill='%23ff4d6d' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='60' y='195' font-size='14' font-weight='bold' fill='%2300cfff' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3Ctext x='118' y='210' font-size='12' font-weight='bold' fill='%23a8ff78' text-anchor='middle' dominant-baseline='middle'%3EX%3C/text%3E%3C/svg%3E");
              background-size: cover;
              border: none;
          }}

          .eyebrow {{
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 176, 0, 0.12);
            color: #ffd15a;
            font-size: 12px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            border: 1px solid rgba(255, 176, 0, 0.18);
          }}

          h1 {{
            margin: 14px 0 12px;
            font-size: clamp(2.4rem, 5vw, 4.3rem);
            line-height: 0.94;
            max-width: 760px;
            color: white;
            letter-spacing: -0.04em;
          }}


          .hero-meta {{
            display: inline-flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 18px;
          }}

          .chip {{
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            color: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
          }}

          .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 14px;
            margin-top: 112px;
            position: relative;
            z-index: 1;
          }}

          .stat {{
            padding: 18px;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.045);
            color: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
          }}

          .stat strong {{
            display: block;
            font-size: 1rem;
            margin-bottom: 6px;
            color: #ffd15a;
            text-transform: uppercase;
            letter-spacing: 0.12em;
          }}

          .layout {{
            display: grid;
            grid-template-columns: 1.25fr 0.95fr;
            gap: 18px;
            margin-top: 18px;
          }}

          .panel {{
            border: 1px solid var(--line);
            border-radius: 24px;
            background: var(--panel);
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
          }}

          .panel::after {{
            content: "";
            position: absolute;
            inset: auto -50px -50px auto;
            width: 130px;
            height: 130px;
            background: radial-gradient(circle, rgba(53, 208, 141, 0.12), transparent 70%);
            pointer-events: none;
          }}

          .panel-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 22px 22px 0;
          }}

          .panel-head h2,
          .panel-head h3 {{
            margin: 0;
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-weight: 700;
            letter-spacing: -0.03em;
          }}

          .panel-body {{
            padding: 22px;
          }}

          .grid {{
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }}

          .full {{
            grid-column: 1 / -1;
          }}

          label {{
            display: block;
            font-size: 0.88rem;
            margin-bottom: 6px;
            color: var(--muted);
          }}

          option {{
            color: black;
          }}

          input, select, textarea, button {{
            width: 100%;
            font: inherit;
          }}

          input, select, textarea {{
            border: 1px solid rgba(213, 243, 229, 0.12);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.05);
            padding: 12px 14px;
            color: var(--ink);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
          }}

          input::placeholder,
          textarea::placeholder {{
            color: #8ea79e;
          }}

          textarea {{
            min-height: 150px;
            resize: vertical;
          }}

          button {{
            border: 0;
            border-radius: 999px;
            padding: 12px 18px;
            cursor: pointer;
            transition: transform 160ms ease, opacity 160ms ease;
          }}

          button:hover {{
            transform: translateY(-1px);
          }}

          button:disabled {{
            opacity: 0.6;
            cursor: wait;
            transform: none;
          }}

          .primary {{
            background: linear-gradient(135deg, #f7a400, var(--accent));
            color: #18261f;
            font-weight: 700;
          }}

          .secondary {{
            background: linear-gradient(135deg, var(--accent-3), #4cc2ff);
            color: white;
            font-weight: 700;
          }}

          .ghost {{
            background: rgba(255, 255, 255, 0.06);
            color: var(--ink);
          }}

          .chat-output {{
            display: flex;
            flex-direction: column;
            gap: 14px;
          }}

          .bubble {{
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--panel-strong);
          }}

          .bubble.user {{
            background: rgba(119, 215, 255, 0.1);
            border-color: rgba(119, 215, 255, 0.16);
          }}

          .bubble.system {{
            background: rgba(53, 208, 141, 0.08);
            border-color: rgba(53, 208, 141, 0.16);
          }}

          .muted {{
            color: var(--muted);
          }}

          .result-card {{
            padding: 16px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.05);
            position: relative;
          }}

          .result-matchup {{
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 8px;
          }}

          .result-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #ffd15a;
          }}

          .result-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ink);
          }}

          .coach-card {{
            display: grid;
            gap: 12px;
            margin-top: 8px;
          }}

          .coach-summary {{
            font-size: 1rem;
            line-height: 1.6;
          }}

          .coach-section-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #ffd15a;
          }}

          .coach-list {{
            margin: 0;
            padding-left: 18px;
            color: var(--ink);
          }}

          .coach-list li {{
            margin: 0 0 6px 0;
          }}

          .metric-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 12px;
          }}

          .metric {{
            padding: 12px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.045);
          }}

          .metric strong {{
            display: block;
            font-size: 1.1rem;
          }}

          .hint-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
          }}

          .hint {{
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.04);
            cursor: pointer;
            color: var(--ink);
          }}

          .side-stack {{
            display: grid;
            gap: 18px;
          }}

          .scoreboard {{
            border: 1px solid var(--line);
            border-radius: 24px;
            background:
              linear-gradient(180deg, rgba(10, 27, 23, 0.98), rgba(18, 42, 36, 0.98));
            padding: 18px;
            box-shadow: var(--shadow);
          }}

          .scoreboard strong {{
            display: block;
            margin-bottom: 8px;
            color: #ffd15a;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.82rem;
          }}

          .score-row {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
          }}

          .score-row:last-child {{
            border-bottom: 0;
          }}

          .score-row span:last-child {{
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
          }}

          .court-mini {{
            margin-top: 12px;
            height: 124px;
            border-radius: 18px;
            background:
              linear-gradient(90deg, transparent 0 8%, var(--court-line) 8% 9%, transparent 9% 91%, var(--court-line) 91% 92%, transparent 92%),
              linear-gradient(180deg, transparent 0 12%, var(--court-line) 12% 13%, transparent 13% 87%, var(--court-line) 87% 88%, transparent 88%),
              linear-gradient(90deg, transparent 0 49.4%, var(--court-line) 49.4% 50.6%, transparent 50.6%),
              linear-gradient(180deg, transparent 0 49.2%, var(--court-line) 49.2% 50.8%, transparent 50.8%),
              linear-gradient(180deg, var(--court) 0%, var(--court-deep) 100%);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(233, 247, 240, 0.12);
          }}

          .court-mini::before {{
            content: "";
            position: absolute;
            left: 50%;
            top: 8px;
            bottom: 8px;
            width: 2px;
            transform: translateX(-50%);
            background: rgba(255, 255, 255, 0.28);
          }}

          .court-mini::after {{
            content: "";
            position: absolute;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            right: 28px;
            top: 26px;
            background: var(--accent);
            box-shadow:
              -54px 24px 0 0 rgba(255, 176, 0, 0.8),
              -108px 52px 0 0 rgba(255, 176, 0, 0.55);
          }}

          pre {{
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
          }}

          @media (max-width: 960px) {{
            .layout {{
              grid-template-columns: 1fr;
            }}
          }}

          @media (max-width: 700px) {{
            .shell {{
              width: min(100% - 20px, 100%);
              margin-top: 10px;
            }}

            .hero, .panel-body {{
              padding: 18px;
            }}

            .hero::after {{
              display: none;
            }}

            .stats {{
              margin-top: 26px;
            }}

            .panel-head {{
              padding: 18px 18px 0;
            }}

            .grid {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <main class="shell">
          <section class="hero">
            <div class="hero-copy">
              <div class="eyebrow">Badminton AI Matchroom</div>
              <h1>Read the court, test the matchup, then coach the next point.</h1>
              <p>
                A cleaner matchday layout for badminton: one place to ask the coach, inspect win probability,
                and trial tactical changes before the next rally starts.
              </p>
              <div class="hero-meta">
                <span class="chip">AI coaching console</span>
                <span class="chip">PAT-backed predictions</span>
                <span class="chip">Strategy drill view</span>
              </div>
            </div>
            <div class="stats">
              <div class="stat">
                <strong>Rally Brief</strong>
                Ask in plain language like a player, analyst, or coach.
              </div>
              <div class="stat">
                <strong>Win Read</strong>
                Check the matchup before stepping onto the court.
              </div>
              <div class="stat">
                <strong>Adjustment Lab</strong>
                Compare tactical levers and their expected impact.
              </div>
            </div>
          </section>

          <section class="layout">
            <div class="panel">
              <div class="panel-head">
                <h2>AI Coach</h2>
                <span class="muted">Rally briefing</span>
              </div>
              <div class="panel-body">
                <form id="chat-form" class="grid">
                  <div class="full">
                    <label for="query">Ask the coach</label>
                    <textarea id="query" name="query" placeholder="Example: Against Kento MOMOTA, should Viktor AXELSEN press the attack early or extend rallies and protect his error rate?"></textarea>
                  </div>
                  <div>
                    <label for="chat-mode">Execution mode</label>
                    <select id="chat-mode" name="mode">
                      <option value="mock">Mock PAT</option>
                      <option value="real">Real PAT</option>
                    </select>
                  </div>
                  <div>
                    <label for="chat-window">Window</label>
                    <input id="chat-window" name="window" type="number" min="1" max="500" value="30" />
                  </div>
                  <div>
                    <label for="chat-budget">Strategy budget</label>
                    <input id="chat-budget" name="budget" type="number" min="1" max="1000" value="60" />
                  </div>
                  <div>
                    <label for="show-trace">Trace output</label>
                    <select id="show-trace" name="show_trace">
                      <option value="false">Hide trace</option>
                      <option value="true">Show trace</option>
                    </select>
                  </div>
                  <div class="full">
                    <button class="primary" id="chat-submit" type="submit">Run AI coach</button>
                  </div>
                </form>

                <div class="hint-list" id="chat-hints">
                  <button class="hint" type="button">Predict Viktor AXELSEN vs Kento MOMOTA</button>
                  <button class="hint" type="button">What strategy should Carolina MARIN use against An Se Young?</button>
                  <button class="hint" type="button">How can PUSARLA V. Sindhu improve her odds versus Ratchanok INTANON?</button>
                </div>

                <div class="chat-output" id="chat-output" style="margin-top: 18px;">
                  <div class="bubble system">
                    AI mode is ready. Ask for a matchup read, a serving idea, or a tactical shift and this view will call the same coach executor used by the CLI.
                  </div>
                </div>
              </div>
            </div>

            <div class="side-stack">
              <div class="panel">
                <div class="panel-head">
                  <h3>Direct Prediction</h3>
                  <span class="muted">Match probability</span>
                </div>
                <div class="panel-body">
                  <form id="predict-form" class="grid">
                    <div>
                      <label for="predict-a">Player</label>
                      <input id="predict-a" name="a" list="players" placeholder="Viktor AXELSEN" />
                    </div>
                    <div>
                      <label for="predict-b">Opponent</label>
                      <input id="predict-b" name="b" list="players" placeholder="Kento MOMOTA" />
                    </div>
                    <div>
                      <label for="predict-mode">Mode</label>
                      <select id="predict-mode" name="mode">
                        <option value="mock">Mock</option>
                        <option value="real">Real</option>
                      </select>
                    </div>
                    <div>
                      <label for="predict-window">Window</label>
                      <input id="predict-window" name="window" type="number" min="1" max="500" value="30" />
                    </div>
                    <div class="full">
                      <button class="secondary" id="predict-submit" type="submit">Compute win probability</button>
                    </div>
                  </form>
                  <div id="predict-result" style="margin-top: 14px;" class="muted">No prediction yet.</div>
                </div>
              </div>

              <div class="panel">
                <div class="panel-head">
                  <h3>Strategy Explorer</h3>
                  <span class="muted">Adjustment board</span>
                </div>
                <div class="panel-body">
                  <form id="strategy-form" class="grid">
                    <div>
                      <label for="strategy-a">Player</label>
                      <input id="strategy-a" name="a" list="players" placeholder="Carolina MARIN" />
                    </div>
                    <div>
                      <label for="strategy-b">Opponent</label>
                      <input id="strategy-b" name="b" list="players" placeholder="An Se Young" />
                    </div>
                    <div>
                      <label for="strategy-mode">Mode</label>
                      <select id="strategy-mode" name="mode">
                        <option value="mock">Mock</option>
                        <option value="real">Real</option>
                      </select>
                    </div>
                    <div>
                      <label for="strategy-window">Window</label>
                      <input id="strategy-window" name="window" type="number" min="1" max="500" value="30" />
                    </div>
                    <div>
                      <label for="strategy-budget">Budget</label>
                      <input id="strategy-budget" name="budget" type="number" min="1" max="1000" value="60" />
                    </div>
                    <div class="full">
                      <button class="primary" id="strategy-submit" type="submit">Find best adjustment</button>
                    </div>
                  </form>
                  <div id="strategy-result" style="margin-top: 14px;" class="muted">No strategy run yet.</div>
                </div>
              </div>
            </div>
          </section>
        </main>

        <datalist id="players"></datalist>

        <script>
          const playerSeed = {example_json};
          const playerList = document.getElementById("players");
          const chatOutput = document.getElementById("chat-output");

          function el(tag, options = {{}}) {{
            const node = document.createElement(tag);
            if (options.className) {{
              node.className = options.className;
            }}
            if (options.text !== undefined) {{
              node.textContent = String(options.text);
            }}
            return node;
          }}

          function escapeHtml(value) {{
            return String(value)
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
          }}

          function formatPercent(value) {{
            const num = Number(value);
            if (!Number.isFinite(num)) {{
              return escapeHtml(value);
            }}
            return `${{(num * 100).toFixed(2)}}%`;
          }}

          function extractJsonPayload(text) {{
            const raw = String(text || "").trim();
            if (!raw) {{
              return null;
            }}

            try {{
              return JSON.parse(raw);
            }} catch (_err) {{
            }}

            const fenced = raw.match(/```json\\s*([\\s\\S]*?)\\s*```/i) || raw.match(/```\\s*([\\s\\S]*?)\\s*```/i);
            if (fenced) {{
              try {{
                return JSON.parse(fenced[1]);
              }} catch (_err) {{
              }}
            }}

            const start = raw.indexOf("{{");
            const end = raw.lastIndexOf("}}");
            if (start >= 0 && end > start) {{
              try {{
                return JSON.parse(raw.slice(start, end + 1));
              }} catch (_err) {{
              }}
            }}

            return null;
          }}

          function formatCoachJson(payload) {{
            if (!payload || typeof payload !== "object") {{
              return null;
            }}

            const advice = payload.advice || payload.summary || payload.answer;
            const details = payload.strategy_details || payload.details || payload.data || null;
            const adjustments = details && Array.isArray(details.key_strategic_adjustments)
              ? details.key_strategic_adjustments
              : [];

            const body = el("div", {{ className: "coach-card" }});
            if (advice) {{
              const section = el("div");
              section.appendChild(el("div", {{ className: "coach-section-title", text: "Coach Advice" }}));
              section.appendChild(el("div", {{ className: "coach-summary", text: advice }}));
              body.appendChild(section);
            }}

            if (details && typeof details === "object") {{
              const metricRow = el("div", {{ className: "metric-row" }});
              if ("improved_win_probability" in details) {{
                const metric = el("div", {{ className: "metric" }});
                metric.appendChild(el("span", {{ className: "muted", text: "Improved win rate" }}));
                metric.appendChild(el("strong", {{ text: formatPercent(details.improved_win_probability) }}));
                metricRow.appendChild(metric);
              }}
              if ("probability_delta" in details) {{
                const metric = el("div", {{ className: "metric" }});
                metric.appendChild(el("span", {{ className: "muted", text: "Delta" }}));
                metric.appendChild(el("strong", {{ text: formatPercent(details.probability_delta) }}));
                metricRow.appendChild(metric);
              }}
              body.appendChild(metricRow);
            }}

            if (adjustments.length) {{
              const section = el("div");
              section.appendChild(el("div", {{ className: "coach-section-title", text: "Key Adjustments" }}));
              const list = el("ul", {{ className: "coach-list" }});
              adjustments.forEach((item) => {{
                const area = item.area || "Adjustment";
                const impact = item.delta_impact ? `: ${{item.delta_impact}}` : "";
                list.appendChild(el("li", {{ text: `${{area}}${{impact}}` }}));
              }});
              section.appendChild(list);
              body.appendChild(section);
            }}

            if (!advice && !adjustments.length) {{
              body.appendChild(el("pre", {{ text: JSON.stringify(payload, null, 2) }}));
            }}

            return body;
          }}

          function formatCoachAnswer(answer) {{
            const parsed = extractJsonPayload(answer);
            if (parsed) {{
              const formatted = formatCoachJson(parsed);
              if (formatted) {{
                return formatted;
              }}
            }}
            return el("div", {{ className: "coach-summary", text: answer }});
          }}

          function addBubble(kind, content) {{
            const bubble = document.createElement("div");
            bubble.className = `bubble ${{kind}}`;
            bubble.append(content);
            chatOutput.prepend(bubble);
          }}

          function renderPrediction(result) {{
            const card = el("div", {{ className: "result-card" }});
            const matchup = el("div", {{ className: "result-matchup" }});
            matchup.appendChild(el("span", {{ className: "result-label", text: "Prediction For" }}));
            matchup.appendChild(el("span", {{ className: "result-title", text: `${{result.player_a}} vs ${{result.player_b}}` }}));
            card.appendChild(matchup);
            card.appendChild(el("div", {{ className: "muted", text: `${{result.player_a}} is the player being evaluated.` }}));

            const metricRow = el("div", {{ className: "metric-row" }});
            const probMetric = el("div", {{ className: "metric" }});
            probMetric.appendChild(el("span", {{ className: "muted", text: `${{result.player_a}} win probability` }}));
            probMetric.appendChild(el("strong", {{ text: `${{(result.probability * 100).toFixed(2)}}%` }}));
            metricRow.appendChild(probMetric);

            const modeMetric = el("div", {{ className: "metric" }});
            modeMetric.appendChild(el("span", {{ className: "muted", text: "Mode" }}));
            modeMetric.appendChild(el("strong", {{ text: result.mode }}));
            metricRow.appendChild(modeMetric);

            card.appendChild(metricRow);
            const artifacts = el("div", {{ className: "muted", text: `Artifacts: ${{result.run_dir}}` }});
            artifacts.style.marginTop = "10px";
            card.appendChild(artifacts);
            return card;
          }}

          function renderStrategy(result) {{
            const best = result.best_candidate;
            const card = el("div", {{ className: "result-card" }});
            const matchup = el("div", {{ className: "result-matchup" }});
            matchup.appendChild(el("span", {{ className: "result-label", text: "Strategy For" }}));
            matchup.appendChild(el("span", {{ className: "result-title", text: `${{result.player_a}} vs ${{result.player_b}}` }}));
            card.appendChild(matchup);
            card.appendChild(el("div", {{ className: "muted", text: `${{result.player_a}} is the player being optimized against ${{result.player_b}}.` }}));

            const metricRow = el("div", {{ className: "metric-row" }});
            const baselineMetric = el("div", {{ className: "metric" }});
            baselineMetric.appendChild(el("span", {{ className: "muted", text: `${{result.player_a}} baseline` }}));
            baselineMetric.appendChild(el("strong", {{ text: `${{(result.baseline_probability * 100).toFixed(2)}}%` }}));
            metricRow.appendChild(baselineMetric);

            const improvedMetric = el("div", {{ className: "metric" }});
            improvedMetric.appendChild(el("span", {{ className: "muted", text: `${{result.player_a}} improved` }}));
            improvedMetric.appendChild(el("strong", {{ text: `${{(result.improved_probability * 100).toFixed(2)}}%` }}));
            metricRow.appendChild(improvedMetric);

            const deltaMetric = el("div", {{ className: "metric" }});
            deltaMetric.appendChild(el("span", {{ className: "muted", text: "Delta" }}));
            deltaMetric.appendChild(el("strong", {{ text: `${{(result.delta * 100).toFixed(2)}}%` }}));
            metricRow.appendChild(deltaMetric);

            card.appendChild(metricRow);

            const description = el(
              "div",
              {{
                className: "muted",
                text:
                  `Best candidate for ${{result.player_a}}: short serve ${{(best.serve_short_delta * 100).toFixed(1)}}%, ` +
                  `attack ${{(best.attack_delta * 100).toFixed(1)}}%, ` +
                  `unforced-error proxy ${{(best.unforced_error_delta * 100).toFixed(1)}}%, ` +
                  `return pressure ${{(best.return_pressure_delta * 100).toFixed(1)}}%.`,
              }},
            );
            description.style.marginTop = "12px";
            card.appendChild(description);

            const artifacts = el("div", {{ className: "muted", text: `Artifacts: ${{result.run_dir}}` }});
            artifacts.style.marginTop = "10px";
            card.appendChild(artifacts);
            return card;
          }}

          async function postJson(url, payload) {{
            const response = await fetch(url, {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify(payload),
            }});
            const data = await response.json();
            if (!response.ok) {{
              throw new Error(data.detail || "Request failed");
            }}
            return data;
          }}

          async function seedPlayers() {{
            playerSeed.forEach((name) => {{
              const option = document.createElement("option");
              option.value = name;
              playerList.appendChild(option);
            }});
            try {{
              const response = await fetch("/players");
              const data = await response.json();
              playerList.innerHTML = "";
              data.players.forEach((name) => {{
                const option = document.createElement("option");
                option.value = name;
                playerList.appendChild(option);
              }});
            }} catch (_err) {{
            }}
          }}

          document.getElementById("chat-form").addEventListener("submit", async (event) => {{
            event.preventDefault();
            const submit = document.getElementById("chat-submit");
            submit.disabled = true;
            const payload = {{
              query: document.getElementById("query").value,
              mode: document.getElementById("chat-mode").value,
              window: Number(document.getElementById("chat-window").value),
              budget: Number(document.getElementById("chat-budget").value),
              show_trace: document.getElementById("show-trace").value === "true",
            }};
            addBubble("user", payload.query);
            try {{
              const result = await postJson("/chat", payload);
              const body = document.createDocumentFragment();
              const heading = el("div");
              heading.appendChild(el("strong", {{ text: "Coach answer" }}));
              body.appendChild(heading);
              body.appendChild(formatCoachAnswer(result.answer));

              const runDir = el("div", {{ className: "muted", text: `Run directory: ${{result.payload.run_dir}}` }});
              runDir.style.marginTop = "10px";
              body.appendChild(runDir);
              if (result.show_trace) {{
                const traceWrap = el("div");
                traceWrap.style.marginTop = "12px";
                traceWrap.appendChild(el("strong", {{ text: "Trace" }}));
                traceWrap.appendChild(el("pre", {{ text: JSON.stringify(result.tool_trace, null, 2) }}));
                body.appendChild(traceWrap);
              }}
              addBubble("system", body);
            }} catch (err) {{
              const body = document.createDocumentFragment();
              body.appendChild(el("strong", {{ text: "Request failed" }}));
              const message = el("div", {{ text: String(err.message || err) }});
              message.style.marginTop = "8px";
              body.appendChild(message);
              addBubble("system", body);
            }} finally {{
              submit.disabled = false;
            }}
          }});

          document.querySelectorAll("#chat-hints .hint").forEach((button) => {{
            button.addEventListener("click", () => {{
              document.getElementById("query").value = button.textContent;
            }});
          }});

          document.getElementById("predict-form").addEventListener("submit", async (event) => {{
            event.preventDefault();
            const submit = document.getElementById("predict-submit");
            submit.disabled = true;
            document.getElementById("predict-result").textContent = "Running prediction...";
            try {{
              const result = await postJson("/predict", {{
                a: document.getElementById("predict-a").value,
                b: document.getElementById("predict-b").value,
                mode: document.getElementById("predict-mode").value,
                window: Number(document.getElementById("predict-window").value),
              }});
              document.getElementById("predict-result").replaceChildren(renderPrediction(result));
            }} catch (err) {{
              document.getElementById("predict-result").textContent = String(err.message || err);
            }} finally {{
              submit.disabled = false;
            }}
          }});

          document.getElementById("strategy-form").addEventListener("submit", async (event) => {{
            event.preventDefault();
            const submit = document.getElementById("strategy-submit");
            submit.disabled = true;
            document.getElementById("strategy-result").textContent = "Running strategy search...";
            try {{
              const result = await postJson("/strategy", {{
                a: document.getElementById("strategy-a").value,
                b: document.getElementById("strategy-b").value,
                mode: document.getElementById("strategy-mode").value,
                window: Number(document.getElementById("strategy-window").value),
                budget: Number(document.getElementById("strategy-budget").value),
              }});
              document.getElementById("strategy-result").replaceChildren(renderStrategy(result));
            }} catch (err) {{
              document.getElementById("strategy-result").textContent = String(err.message || err);
            }} finally {{
              submit.disabled = false;
            }}
          }});

          seedPlayers();
        </script>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _render_home()


@app.get("/players")
def players() -> dict[str, list[str]]:
    names = sorted(service.adapter.players_df["name"].astype(str).tolist())
    return {"players": names}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        result = service.predict(player_a=req.a, player_b=req.b, window=req.window, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "probability": result.probability,
        "mode": result.mode,
    }


@app.post("/strategy")
def strategy(req: StrategyRequest) -> dict[str, Any]:
    try:
        result = service.strategy(
            player_a=req.a,
            player_b=req.b,
            window=req.window,
            mode=req.mode,
            budget=req.budget,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "baseline_probability": result.baseline_probability,
        "improved_probability": result.improved_probability,
        "delta": result.delta,
        "best_candidate": result.best_candidate.__dict__,
        "top_alternatives": [cand.__dict__ for cand in result.top_alternatives],
        "mode": result.mode,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        result = chat_executor.run(
            req.query,
            mode=req.mode,
            window=req.window,
            budget=req.budget,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "answer": result.answer,
        "plan": result.plan.model_dump(),
        "payload": result.payload,
        "tool_trace": result.tool_trace,
        "show_trace": req.show_trace,
    }
