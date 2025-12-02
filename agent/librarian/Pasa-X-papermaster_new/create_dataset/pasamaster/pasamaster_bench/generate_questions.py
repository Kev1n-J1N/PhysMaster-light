import json
import requests
from typing import List, Dict, Optional, Tuple, Any
import random
from dataclasses import dataclass
import arxiv
from prompt import Paper, JSONResponseExtractor, DESCRIPTIVE_EXTRACTOR_PROMPT, DESCRIPTIVE_GENERATOR_PROMPT, COMPOUND_GENERATOR_PROMPT


@dataclass
class BenchmarkEntry:
    id: str
    papers: List[Dict]
    keywords: List[str]
    queries: List[str]
    question: str
    question_type: str
    difficulty: str
    metadata: Dict

# ==================== Paper Keyword Extractor ====================

class Extractor:
    """Extract keywords and generate queries from papers using LLM or crawler"""

    def __init__(self, llm, use_llm: bool = True, question_type: str = "descriptive"):
        self.client = llm
        self.use_llm = use_llm
        self.question_type = question_type  # "descriptive" or "compound"

    async def extract_keywords_and_queries(self, paper: Dict) -> Tuple[List[str], List[str]]:
        """Extract keywords and generate relational queries from a paper"""
        paper = self._dict_to_paper(paper)
        if self.use_llm:
            return await self._llm_extraction(paper)
        else:
            return await self._crawler_extraction(paper)

    def _dict_to_paper(self, paper_dict: Dict) -> Paper:
        """Convert dictionary to Paper dataclass"""
        return Paper(
            id=paper_dict.get('id', ''),
            title=paper_dict.get('title', ''),
            abstract=paper_dict.get('abstract', ''),
            authors=paper_dict.get('authors', []),
            year=paper_dict.get('year', 0),
            venue=paper_dict.get('venue', ''),
            doi=paper_dict.get('doi'),
            url=paper_dict.get('url'),
            keywords=paper_dict.get('keywords'),
            sections=paper_dict.get('sections', [])
        )

    async def _llm_extraction(self, paper: Paper) -> Tuple[List[str], List[str]]:
        """Use LLM to extract keywords and generate queries"""
        if self.question_type == "descriptive":
            prompt = DESCRIPTIVE_EXTRACTOR_PROMPT.format(paper=paper)

        try:
            response = self.client.generate(
                prompt=prompt)
            
            extractor = JSONResponseExtractor()
            result = extractor.extract_json_from_response(response)

            if not extractor.validate_extraction_format(result, ["keywords", "queries"]):
                print("[ERROR] 提取的JSON格式不正确")
                return [], []

            return result["keywords"], result["queries"]
            
        except Exception as e:
            print(f"LLM extraction failed: {e}")
    
    async def _crawler_extraction(self, paper: Paper) -> Tuple[List[str], List[str]]:
        """Crawl paper webpage to extract keywords"""
        pass
        

class QuestionGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_descriptive_questions(self, 
                                      papers: List[Paper],
                                      keywords: List[str],
                                      queries: List[str],
                                      ) -> List[Dict]:
        """Generate ONE scenario-based questions without explicit keywords"""
        context = self._format_papers_context(papers)
        prompt = DESCRIPTIVE_GENERATOR_PROMPT.format(
                context=context,
                hidden_concepts=', '.join(queries + keywords),
                queries=queries,
                keywords=keywords
            )

        response = self.llm.generate(prompt)
        try:
            result = json.loads(response)
            return result["questions"][0]["question"]
        except Exception as e:
            print("Error generating descriptive questions:", e)
            return self._generate_fallback_descriptive_questions(papers, keywords)
    
    def generate_compound_question(self,
                                   papers: List[Paper]
                                   ) -> List[Dict]:
        """Generate ONE question with explicit keywords"""
        
        
        prompt = COMPOUND_GENERATOR_PROMPT.format(
            papers = papers
        )

        response = self.llm.generate(prompt)
        try:
            extractor = JSONResponseExtractor()
            result = extractor.extract_json_from_response(response)

            if not extractor.validate_extraction_format(result, ["questions"]):
                print("[ERROR] 提取的JSON格式不正确")
                return []

            return result["questions"]
        except:
            print("Error generating compound questions")

    def _dict_to_paper(self, paper_dict: Dict) -> Paper:
        """Convert dictionary to Paper dataclass"""
        return Paper(
            id=paper_dict.get('id', ''),
            title=paper_dict.get('title', ''),
            abstract=paper_dict.get('abstract', ''),
            authors=paper_dict.get('authors', []),
            year=paper_dict.get('year', 0),
            venue=paper_dict.get('venue', ''),
            doi=paper_dict.get('doi'),
            url=paper_dict.get('url'),
            keywords=paper_dict.get('keywords')
        )
    
    def _format_papers_context(self, papers: List[Paper]) -> str:
        """Format paper information for LLM context"""
        papers = [self._dict_to_paper(p) for p in papers]
        context_parts = []
        for i, paper in enumerate(papers[:5], 1):  # Limit to 5 papers
            context_parts.append(f"""Paper {i}:
Title: {paper.title}
Year: {paper.year}
Abstract: {paper.abstract[:200]}...
Keywords: {', '.join(paper.keywords[:10]) if paper.keywords else 'N/A'}""")
        
        return "\n\n".join(context_parts)
    


class Selector:
    def __init__(self, llm=None):
        self.llm = llm
    
    def select_queries(self, extract_data: List[Dict[str, Any]], num_select: int = 3, num_iter: int = 5) -> List[Dict[str, Any]]:
        """
        Select num_iter entries, each containing num_select DIFFERENT queries 
        (each query from a different paper).
        
        Args:
            extract_data: List of dicts with paper_id, title, keywords, and queries
            num_select: Number of different queries (from different papers) in each entry
            num_iter: Number of entries to return
            
        Returns:
            List of num_iter entries, each containing num_select different queries
        """
        selected_data = []
        
        # Create a pool of all valid queries with their paper info
        all_valid_queries = []
        for paper in extract_data:
            for query in paper["queries"]:
                matching_kw = [kw for kw in paper["keywords"] 
                             if kw.lower() in query.lower()]
                if matching_kw:
                    all_valid_queries.append({
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "keywords": paper["keywords"],
                        "query": query,
                        "matching_keywords": matching_kw
                    })
        
        # Generate num_iter entries
        for i in range(num_iter):
            # For each entry, select num_select queries from different papers
            used_paper_ids = set()
            paper_ids = []
            titles = []
            all_keywords = []
            queries = []
            matching_keywords_list = []
            
            # Shuffle for random selection
            random.shuffle(all_valid_queries)
            
            for query_info in all_valid_queries:
                # Stop when we have enough different queries
                if len(queries) >= num_select:
                    break
                
                # Ensure each query comes from a different paper
                if query_info["paper_id"] not in used_paper_ids:
                    paper_ids.append(query_info["paper_id"])
                    titles.append(query_info["title"])
                    all_keywords.extend(query_info["keywords"])
                    queries.append(query_info["query"])
                    matching_keywords_list.extend(query_info["matching_keywords"])
                    
                    # Mark this paper as used for this entry
                    used_paper_ids.add(query_info["paper_id"])
            
            # Create entry with num_select different queries
            if len(queries) == num_select:
                selected_data.append({
                    "paper_ids": paper_ids,
                    "titles": titles,
                    "keywords": list(set(all_keywords)),  # Remove duplicates
                    "queries": queries,  # num_select different queries
                    "matching_keywords": list(set(matching_keywords_list))  # Remove duplicates
                })
        
        return selected_data