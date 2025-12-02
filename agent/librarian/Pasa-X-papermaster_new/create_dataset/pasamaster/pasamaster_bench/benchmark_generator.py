import os
import json
import random
from typing import List, Dict
import asyncio

from generate_questions import QuestionGenerator, Extractor, Selector

class BenchmarkGenerator:
    """Generates benchmark entries from sample papers"""
    
    def __init__(self, llm: None, 
                 paper_path: str = "papers.json", 
                 base_path: str = "output",
                 question_type: str = "all",
                 max_concurrent_extractions: int = 10):
        self.llm = llm
        self.paper_path = paper_path
        self.sample_papers = []
        self.question_type = question_type  # "descriptive", "compound", or "all"
        self.max_concurrent_extractions = max_concurrent_extractions

        # Load sample papers from json file
        if os.path.exists(paper_path):
            if paper_path.endswith(".json"):
                with open(paper_path, "r", encoding='utf-8') as f:
                    self.sample_papers = json.load(f)
            elif paper_path.endswith("jsonl"):
                with open(paper_path, "r") as f:
                    for line in f:
                        self.sample_papers.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Paper file {paper_path} not found.")

        self.output_dir_path = base_path
        if not os.path.exists(self.output_dir_path):
            os.makedirs(self.output_dir_path)
   

    async def extract_keywords_and_queries(self, paper, extractor: Extractor) -> Dict:
        """Extract keywords and queries for a single paper"""
        try:
            keywords, queries = await extractor.extract_keywords_and_queries(paper)
            return {
                "paper_id": paper["id"],
                "title": paper["title"],
                "keywords": keywords,
                "queries": queries,
                "status": "success"
            }
        except Exception as e:
            print(f"Error extracting from paper {paper.get('id', 'unknown')}: {e}")
            return {
                "paper_id": paper.get("id", "unknown"),
                "title": paper.get("title", "unknown"),
                "keywords": [],
                "queries": [],
                "status": "error",
                "error": str(e)
            }
    
    async def extract_keywords_batch(self, papers: List[Dict], extractor: Extractor) -> List[Dict]:
        """Extract keywords from a batch of papers with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.max_concurrent_extractions)
        
        async def extract_with_semaphore(paper):
            async with semaphore:
                return await self.extract_keywords_and_queries(paper, extractor)
        
        tasks = [extract_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert them to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "paper_id": papers[i].get("id", "unknown"),
                    "title": papers[i].get("title", "unknown"),
                    "keywords": [],
                    "queries": [],
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results

    async def generate_benchmark_entry(self) -> Dict:
        if self.question_type == "descriptive":
            output =  await self.generate_descriptive_benchmark_entry()
        elif self.question_type == "compound":
            output = await self.generate_compound_benchmark_entry()
        else:
            print("Invalid question type, please select either 'descriptive' or 'compound'.")
            return
        
        benchmark_entry = {
            "papers": self.sample_papers,
            "queries": output["queries"],
            "questions": output["questions"]
        }
        
        benchmark_entry_path = os.path.join(self.output_dir_path, "benchmark_entry.json")
        with open(benchmark_entry_path, "w") as f:
            json.dump(benchmark_entry, f, indent=2)
        
        print(f"Saved benchmark entry to {benchmark_entry_path}")
        return benchmark_entry
    
    async def generate_compound_benchmark_entry(self) -> Dict:
        print("Generating questions...")
        question_generator = QuestionGenerator(self.llm)
        compound_questions = []
        extracted_queries = []
        for paper in self.sample_papers:
            extracted_queries.append(paper.get("description", ""))

        questions = question_generator.generate_compound_question(papers = self.sample_papers)
        compound_questions.extend(questions)
        return {
            "queries": extracted_queries,
            "questions": compound_questions
        }

    async def generate_descriptive_benchmark_entry(self) -> Dict:

        # Step 1: Extract keywords using LLM
        print("Step 1: Extracting keywords and queries from papers...")
        extract_data = []
        extractor = Extractor(llm=self.llm, question_type=self.question_type)
        batch_size = 10  # Adjust based on your needs
        extract_data = []
        for i in range(0, len(self.sample_papers), batch_size):
            batch = self.sample_papers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(self.sample_papers) + batch_size - 1)//batch_size}")
            batch_results = await self.extract_keywords_batch(batch, extractor)
            extract_data.extend(batch_results)
        tasks = [self.extract_keywords_and_queries(paper, extractor) for paper in self.sample_papers]
        extract_data = await asyncio.gather(*tasks)

        # save extracted keywords and queries to json file
        keywords_queries_path = os.path.join(self.output_dir_path, "extracted_keywords_queries.json")
        with open(keywords_queries_path, "w") as f:
            json.dump(extract_data, f, indent=2)
        
        with open(keywords_queries_path, "r") as f:
            extract_data = json.load(f)
        
        print(f"Extracted keywords and queries for {len(extract_data)} papers")

        
        # Step 2: Define search queries 
        print("\nStep 2: Selecting search queries...")
        selector = Selector(llm=None)
        # num_select 1,2,4,6,8, num_iter 5
        selected_data = []
        for i in [1, 2, 4, 6, 8]:
            item = selector.select_queries(extract_data, num_select=i, num_iter=1)
            selected_data.extend(item)

        print(f"Selected {len(selected_data)} query sets from extracted data")
        
        # Save selected queries to file
        selected_queries_path = os.path.join(self.output_dir_path, "selected_queries.json")
        with open(selected_queries_path, "w") as f:
            json.dump(selected_data, f, indent=2)
        
        with open(selected_queries_path, "r") as f:
            selected_data = json.load(f)

        # selected_data = random.sample(selected_data, 2)  # Randomly select 2 query sets
        print(f"Loaded {len(selected_data)} selected query sets from file")
        
        
        # Step 3: Generate  questions
        print("\nStep 3: Generating questions...")
        descriptive_questions = []
        extracted_keywords = []
        extracted_queries = []

        for item in selected_data:
            extracted_keywords = item["matching_keywords"]
            extracted_queries = item["queries"]

            question_generator = QuestionGenerator(self.llm)
            
            descriptive_question = question_generator.generate_descriptive_questions(
                self.sample_papers, extracted_keywords, extracted_queries)
            des_data = {
                "question": descriptive_question,
                "keywords": extracted_keywords,
                "queries": extracted_queries
            }
            descriptive_questions.append(des_data)
            print(f"Generated {len(descriptive_questions)} descriptive questions")

        # Save questions to files
        # des_questions_path = os.path.join(self.output_dir_path, "descriptive_questions.json")
        # with open(des_questions_path, "w") as f:
        #     json.dump(descriptive_questions, f, indent=2)
        # print(f"Saved {len(descriptive_questions)} descriptive questions")

        return {
            "queries": extracted_queries,
            "questions": descriptive_questions
        }
        
       

