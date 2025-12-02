import json
import os
from utils.gpt5_utils import call_model_wo_tools


LLM_SYSTEM_PROMPT = """
    You are an scientific information extraction expert. Your task is to convert the user's physics problem into a structured form.

    Follow these instructions carefully:
    - Your output MUST be valid JSON only that strictly follows the given schema.
    - You should understand the core of this physics with scientific rigor, identify the essential physics as well as details likr conserved quantities, symmetries, scales etc.
    - The information you provide should be primarily based on the user's query.
    - If some information has not been mentioned and is considered irrelevant, fill with "None".
    - To eliminate ambiguity or clarify detail, reasonable inferences are allowed based on reliable physics knowledge.
    - Devide the task into several reasonable sub-tasks. To avoid redundant sub-tasks, only make necessary divisions. When the task is simple, only one sub-task is allowed.
    - The number of the items in "related_knowledge" should be less than or equal to {max_keys}
    """

USER_INPUT_PROMPT = """
    User query: "{user_query}"
    Schema to follow strictly: {schema_json}
    Now output ONLY the JSON object that adheres to the above schema.
    """

class Clarifier:
    def __init__(self,config):
        self.schema_file = config.get("schema_file_path")
        self.max_keys = config.get("max_key_concpets",5)

    def task_spec(self, user_query):

        if os.path.exists(os.path.join(self.schema_file, "template.json")):
            task_schema_file = os.path.join(self.schema_file, "template.json")
        else:    
            raise FileNotFoundError(f"No template.json found in {self.schema_file}")

        abs_path = os.path.abspath(task_schema_file)
        with open(task_schema_file, 'r', encoding="utf-8") as f:
            schema = json.load(f)


        system_prompt = LLM_SYSTEM_PROMPT.format(
            max_keys = self.max_keys
        )
        user_prompt = USER_INPUT_PROMPT.format(
                user_query = user_query,
                schema_json = json.dumps(schema,indent=2)
            )
        
        response = call_model_wo_tools(system_prompt=system_prompt, user_prompt=user_prompt)

        return response

    def _parse_result(self, result):
        """Parse the LLM response into structured format"""
        try:
            # Try to extract JSON from the response
            if "{" in result and "}" in result:
                start_idx = result.find("{")
                end_idx = result.rfind("}") + 1
                json_str = result[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"raw_output": result, "error": "No structured output found"}
        except Exception as e:
            return {
                "raw_output": result, 
                "error": f"Failed to parse response: {str(e)}"
            }

    def run(self, raw_input):
        result = self.task_spec(raw_input)
        contract = self._parse_result(result)
        return contract
