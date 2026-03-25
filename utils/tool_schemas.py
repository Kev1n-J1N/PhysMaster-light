from __future__ import annotations


PYTHON_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "Python_code_interpreter",
        "description": "Execute python code and return the stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python script to execute.",
                }
            },
            "required": ["code"],
        },
    },
}


LOAD_SKILL_SPECS_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill_specs",
        "description": (
            "Load installed Codex-style skill files by skill name from SKILL.md. "
            "The returned text provides authoritative workflow guidance and should be treated as the primary source of truth for the selected skill(s). "
            "Returns plain text that contains a [SKILL FULL] header and one or more <SKILL_FULL> blocks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Non-empty list of installed skill names, e.g. ['openai-docs'].",
                }
            },
            "required": ["skill_names"],
            "additionalProperties": False,
        },
    },
}


LIBRARY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "library_search",
        "description": "Search external web/library sources for relevant references and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "description": "Number of results to return.", "default": 5},
            },
            "required": ["query"],
        },
    },
}


LIBRARY_PARSE_TOOL = {
    "type": "function",
    "function": {
        "name": "library_parse",
        "description": "Read and analyze a specific web page or PDF link for the current question.",
        "parameters": {
            "type": "object",
            "properties": {
                "link": {"type": "string", "description": "Web page or PDF URL."},
                "user_prompt": {"type": "string", "description": "Question about the linked content."},
                "llm": {"type": "string", "description": "Optional model override."},
            },
            "required": ["link", "user_prompt"],
        },
    },
}


PRIOR_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "prior_search",
        "description": "Search LANDAU prior knowledge base for relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return.",
                    "default": 3,
                },
                "expand_context": {
                    "type": "boolean",
                    "description": "Include prev/next chunks for context.",
                    "default": False,
                },
                "return_format": {
                    "type": "string",
                    "description": "Return text for prompt or raw JSON.",
                    "enum": ["text", "json"],
                    "default": "text",
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source file filters, e.g. ['bonnerot_2022.pdf'].",
                },
                "chapter": {
                    "type": "string",
                    "description": "Optional chapter filter.",
                },
                "section_prefix": {
                    "type": "string",
                    "description": "Optional section prefix filter, e.g. '2.1'.",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional keyword filters.",
                },
                "rewrite_query": {
                    "type": "boolean",
                    "description": "Whether to apply query rewrite/expansion before retrieval.",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
}


THEORETICIAN_CORE_TOOLS = [PYTHON_CODE_TOOL, LOAD_SKILL_SPECS_TOOL]
LIBRARY_TOOLS = [LIBRARY_SEARCH_TOOL, LIBRARY_PARSE_TOOL]
