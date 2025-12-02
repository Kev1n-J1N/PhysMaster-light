import openai


class LLMClient:
    """Wrapper for LLM API calls"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", api_base = None):
        if api_base:
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates structured data for paper search benchmarks."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

