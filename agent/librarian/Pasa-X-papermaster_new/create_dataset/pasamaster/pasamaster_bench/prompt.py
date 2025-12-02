from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import json
import re

@dataclass
class Paper:
    id: str
    title: str
    abstract: Optional[str]
    authors: List[str]
    year: int
    venue: str
    doi: Optional[str] = None
    url: Optional[str] = None
    keywords: Optional[List[str]] = None
    sections: Optional[List[Dict[str, str]]] = None
    description: Optional[str] = None





COMPOUND_GENERATOR_PROMPT = """
You will receive information about several papers. For each paper, generate a unique search question that can ONLY be answered by that specific target paper.

Input Information:
1. Multiple paper entries, each containing: title, authors, publication date, abstract, cited_by, description. {papers}
2. For each paper, a "cited_by" field containing papers that have cited it, along with their "description" content which represents how they comment on and characterize thi paper when citing it

Requirements:
1. Analyze citation contexts: Examine how these papers describe and contextualize each other in their citations. These descriptions reveal each paper's key contributions, methodologies, and significance in the field.
2. Create one question per paper: Process each paper independently and generate a distinct question for it. Each output must include:
   - paper_id: Clearly identify which paper the question targets
   - question: A compound query uniquely identifying this paper
3. Ensure unique answerability: The question must describe details so specific that ONLY the target paper can satisfy all criteria. Achieve this by:
   - Preserving specific details from the target paper's description - not just naming techniques or topics, but describing HOW they work, WHAT problems they solve, or WHY they're significant
   - Integrating multiple citation comments: Combine 2-5 specific details from how different papers comment on and characterize the target paper when citing it
   - Creating logical intersections: Connect these details in ways that form a unique fingerprint (e.g., "paper that does X while also addressing Y and being cited for Z")
   - Selectively leveraging author and publication details: Selectively include author collaborations and published date info (e.g., "from OpenAI with over 200 authors"), but only if it has unique features. DO NOT include temporal information.
4. Example of Good vs Bad Questions:
    - Bad: "What paper discusses deep learning and spiking neural networks and backpropagation?"
    - Good: "What paper reviews methods for training multilayer spiking networks that addresses the non-differentiable transfer function challenge and compares computational costs while showing the accuracy gap with traditional ANNs is decreasing?"

Respond with ONLY a valid JSON object in the following exact format, Start your response with ```json, End your response with ```, Include exactly the key "questions", Each question must have the keys "question", "question_type", "keywords", and "queries":
{{
    "questions": [
        {{
            "question": "The question with compound query",
            "paper_id": "the unique identifier for the target paper"
        }}
    ]
}}"""


DESCRIPTIVE_EXTRACTOR_PROMPT =  """
Analyze this research paper and extract information:
        
        Title: {paper.title}
        Abstract: {paper.abstract}
        Content: {paper.sections}
    
        Task 1: Extract FIVE important technical keywords/phrases from this paper.
        [Focus on]: methods, algorithms, techniques, applications, problems solved.

        Task 2: Generate FIVE relational research questions that researchers might have posed when reading this text
        [Requirements:]
        1. Construct the question-answer pairs based on the text, the answer should be the source paper.
        2. Do not ask questions including "or" or "and" that may involve more than one condition.
        3. Clarity: Formulate questions clearly and unambiguously to prevent confusion. A query is ambiguous if there is insufficient context and additional information is needed. For instance, abbreviations with multiple meanings can create ambiguity, leading to the corresponding paper being incomplete answer lists. An answer paper is considered qualified if it aligns with the requirements of the query. The paper should address all or most of the essential factors that make it a suitable response.
        4. Contextual Definitions: Include explanations or definitions for specialized terms and concepts used in the questions

        Output format:
        {{
            "keywords": ["keyword1", "keyword2", ...],
            "queries": ["query1", "query2", ...]
        }}
"""

DESCRIPTIVE_GENERATOR_PROMPT = """
Based on these research papers, generate ONE realistic scenario-based questions that researchers or engineers might ask when searching for similar papers., No additional text before or after the JSON block

Papers context:
{context}

Hidden concepts (do NOT use these directly in questions):
{hidden_concepts}

Requirements for questions:
1. Describe a real problem or scenario without using technical keywords directly
2. Focus on practical challenges, implementation issues, or research needs  
3. Make it natural - like someone describing their problem without knowing the exact technical terms
4. Each question should implicitly relate to the paper themes but not mention them explicitly

`Output format:
{{
    "questions": [
        {{
            "question": "The actual question text",
            "scenario_type": "engineering_problem|research_challenge|implementation_issue|comparison_need",
            "queries":{queries},
            "keywords": {keywords},
            "difficulty": "easy|medium|hard"
        }}
    ]
}}`"""

class JSONResponseExtractor:
    """
    用于从LLM响应中提取JSON格式数据的工具类
    """
    
    def __init__(self):
        # JSON代码块模式
        self.json_block_pattern = r'```json\s*(.*?)\s*```'
        # 直接JSON模式
        self.json_pattern = r'\{.*?\}'
        
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        从响应中提取JSON数据
        
        Args:
            response: LLM的原始响应文本
            
        Returns:
            解析后的JSON字典，失败返回None
        """
        # 方法1: 尝试提取```json代码块
        json_match = re.search(self.json_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 方法2: 查找第一个完整的JSON对象
            json_match = re.search(self.json_pattern, response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        
        # 尝试解析JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON解析失败: {e}")
            print(f"原始JSON字符串: {json_str}")
            return None
    
    def validate_extraction_format(self, data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        验证提取的JSON是否包含必需的键
        
        Args:
            data: 解析后的JSON数据
            required_keys: 必需的键列表
            
        Returns:
            验证是否通过
        """
        if not isinstance(data, dict):
            return False
            
        for key in required_keys:
            if key not in data:
                print(f"[WARNING] 缺少必需的键: {key}")
                return False
                
            # 检查键对应的值是否为列表
            if not isinstance(data[key], list):
                print(f"[WARNING] 键 '{key}' 的值不是列表类型")
                return False
                
        return True
