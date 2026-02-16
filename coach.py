"""
Stage 3: AI Coach — Gemini LLM wrapper with tool-use (function-calling).

Architecture:
    User query (natural language)
      -> Gemini chat completion with tool definitions
      -> LLM decides which tools to call
      -> Tools execute pipeline stages (preprocess, PCSP gen, PAT)
      -> LLM synthesizes results into coaching advice

Uses the Google GenAI SDK with manual tool-use loop.
Tools are inline methods on the AICoach class.
"""
import json
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from config import SportConfig
from preprocess import DataLoader, ParameterExtractor
from pcsp_generator import PCSPGenerator
from pat_runner import PATRunner


# ---------------------------------------------------------------------------
# Tool definitions for Gemini function-calling
# ---------------------------------------------------------------------------

TOOLS = types.Tool(function_declarations=[
    {
        "name": "analyze_matchup",
        "description": (
            "Preprocess historical data for a matchup between two entities "
            "(players/teams). Extracts frequency parameters from historical "
            "data and generates a PCSP model file for PAT verification. "
            "Returns parameter summary and generated file path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity1": {
                    "type": "string",
                    "description": "Name of the first player/team"
                },
                "entity2": {
                    "type": "string",
                    "description": "Name of the second player/team"
                },
                "date": {
                    "type": "string",
                    "description": "Match date in YYYY-MM-DD format"
                },
                "variant": {
                    "type": "string",
                    "description": (
                        "Model variant name (e.g. 'RH_RH' for tennis). "
                        "If omitted, the first configured variant is used."
                    )
                }
            },
            "required": ["entity1", "entity2", "date"]
        }
    },
    {
        "name": "get_win_probability",
        "description": (
            "Get the win probability from a PAT model check result. "
            "If a probability value is provided (from manual PAT run), "
            "records it. Otherwise returns instructions for running PAT."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity1": {
                    "type": "string",
                    "description": "Name of entity 1"
                },
                "entity2": {
                    "type": "string",
                    "description": "Name of entity 2"
                },
                "probability": {
                    "type": "number",
                    "description": "Manually entered probability (0-1) from PAT output"
                }
            },
            "required": ["entity1", "entity2"]
        }
    },
    {
        "name": "compare_parameters",
        "description": (
            "Compare the extracted statistical parameters between two entities. "
            "Shows which entity is stronger in each parameter group "
            "(e.g. serve accuracy, return winners, rally consistency)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity1": {
                    "type": "string",
                    "description": "Name of entity 1"
                },
                "entity2": {
                    "type": "string",
                    "description": "Name of entity 2"
                }
            },
            "required": ["entity1", "entity2"]
        }
    }
])

SYSTEM_PROMPT = """You are an AI sports coach powered by formal methods. You combine:
1. Historical statistical analysis (frequency counts from match data)
2. Probabilistic model checking (PAT/PCSP# formal verification)
3. Sports expertise (tactical interpretation of the numbers)

When a user asks about a matchup, strategy, or prediction:
- Use the analyze_matchup tool to extract parameters from historical data
- Use compare_parameters to break down strengths and weaknesses
- Use get_win_probability to get formal verification results
- Synthesize everything into actionable coaching advice

Always explain your reasoning. Mention when data is limited.
The probabilities come from a formally verified model, not guesswork.

Sport: {sport_name}
Entity type: {entity_name}
Available model variants: {variants}
"""


class AICoach:
    """Gemini-powered coaching assistant with tool-use."""

    def __init__(
        self,
        config: SportConfig,
        gemini_api_key: str,
        model: str = "gemini-2.0-flash",
        pat_path: Optional[str] = None,
        output_dir: str = './output'
    ):
        self.config = config
        self.client = genai.Client(api_key=gemini_api_key)
        self.model = model
        self.output_dir = output_dir

        # Pipeline components
        self.data_loader = DataLoader(config)
        self.extractor = ParameterExtractor(config)
        self.generator = PCSPGenerator()
        self.pat = PATRunner(pat_path)

        # Cache for extracted params (avoid reprocessing)
        self._param_cache: Dict[str, Dict] = {}

        # Conversation history
        self.contents: List[types.Content] = []

        # Build system prompt
        variant_names = [v.name for v in config.variants]
        system_prompt = SYSTEM_PROMPT.format(
            sport_name=config.sport_name,
            entity_name=config.entity_name,
            variants=', '.join(variant_names)
        )

        # Gemini generation config with system instruction and tools
        self.generate_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[TOOLS],
        )

    # -------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Dispatch a tool call to the appropriate handler."""
        handlers = {
            'analyze_matchup': self._tool_analyze_matchup,
            'get_win_probability': self._tool_get_win_probability,
            'compare_parameters': self._tool_compare_parameters,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        return handler(**arguments)

    def _tool_analyze_matchup(
        self,
        entity1: str,
        entity2: str,
        date: str,
        variant: Optional[str] = None
    ) -> str:
        """Run preprocessing + PCSP generation for a matchup."""
        try:
            df = self.data_loader.load()

            # Select variant
            if variant:
                variant_config = self.config.get_variant(variant)
            else:
                variant_config = self.config.variants[0]

            params, metadata = self.extractor.get_all_params(
                df, entity1, entity2, date, variant_config
            )

            # Generate PCSP file
            pcsp_path = self.generator.generate(
                self.config, params, variant_config.name,
                entity1, entity2, date, self.output_dir
            )

            # Cache results
            cache_key = f"{entity1}_vs_{entity2}_{date}"
            self._param_cache[cache_key] = {
                'params': params,
                'metadata': metadata,
                'pcsp_path': pcsp_path,
                'variant': variant_config.name,
            }

            result = {
                'status': 'success',
                **metadata,
                'pcsp_file': pcsp_path,
                'note': 'PCSP file generated. Run PAT to get win probability.',
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    def _tool_get_win_probability(
        self,
        entity1: str,
        entity2: str,
        probability: Optional[float] = None
    ) -> str:
        """Get or record win probability from PAT."""
        if probability is not None:
            return json.dumps({
                'entity1': entity1,
                'entity2': entity2,
                'p1_win_prob': probability,
                'p2_win_prob': round(1 - probability, 4),
                'source': 'manual_pat_result',
            })

        # Look for cached PCSP file to give PAT instructions
        matching = [
            v for k, v in self._param_cache.items()
            if entity1 in k and entity2 in k
        ]
        if matching:
            pat_result = self.pat.run(matching[0]['pcsp_path'])
            return json.dumps(pat_result)

        return json.dumps({
            'status': 'not_found',
            'message': (
                f'No analysis found for {entity1} vs {entity2}. '
                f'Run analyze_matchup first.'
            ),
        })

    def _tool_compare_parameters(
        self,
        entity1: str,
        entity2: str
    ) -> str:
        """Compare per-group parameter breakdowns between entities."""
        matching = [
            v for k, v in self._param_cache.items()
            if entity1 in k and entity2 in k
        ]
        if not matching:
            return json.dumps({
                'status': 'not_found',
                'message': 'Run analyze_matchup first.',
            })

        cached = matching[0]
        params = cached['params']
        variant_config = self.config.get_variant(cached['variant'])

        # Split params into entity1 and entity2 halves
        n = len(params) // 2
        p1_params = params[:n]
        p2_params = params[n:]

        comparison = []
        idx = 0
        for group in variant_config.parameter_groups:
            n_outcomes = len(group.count_queries)
            g1 = p1_params[idx:idx + n_outcomes]
            g2 = p2_params[idx:idx + n_outcomes]
            comparison.append({
                'group': group.name,
                'entity1_counts': g1,
                'entity2_counts': g2,
                'entity1_total': sum(g1),
                'entity2_total': sum(g2),
                'outcomes': list(group.count_queries.keys()),
            })
            idx += n_outcomes

        return json.dumps({
            'entity1': entity1,
            'entity2': entity2,
            'comparison': comparison,
        })

    # -------------------------------------------------------------------
    # Chat interface
    # -------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message and get a coaching response.

        Main loop:
          1. Add user message to conversation
          2. Call Gemini with tool definitions
          3. If LLM wants to call tools -> execute them, feed results back
          4. Repeat until LLM produces a final text response

        Args:
            user_message: Natural language query

        Returns:
            Coach's text response
        """
        self.contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        while True:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.contents,
                config=self.generate_config,
            )

            # Add model response to history
            model_content = response.candidates[0].content
            self.contents.append(model_content)

            # Check for function calls
            function_calls = response.function_calls
            if not function_calls:
                return response.text or ""

            # Execute each function call and build responses
            function_response_parts = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args)
                result = self._execute_tool(fn_name, fn_args)

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result},
                    )
                )

            # Add all function responses as a single user-role Content
            self.contents.append(
                types.Content(role="user", parts=function_response_parts)
            )

    def reset(self):
        """Clear conversation history and param cache."""
        self.contents.clear()
        self._param_cache.clear()
