# agents/question_extractor.py
from typing import Dict, List, Any
from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex
from utils.ocr_gemini import gemini_extract_question_text
from .base_agent import BaseAgent, AgentResult

class QuestionExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__("QuestionExtractor", ["openai_vision", "gemini_vision"])
        self.question_prompt = '''EXTRACT EXAMINATION QUESTIONS WITH COMPLETE DETAILS

You are analyzing an academic examination paper. Extract ALL questions with their complete text, sub-parts, and formatting.

CRITICAL REQUIREMENTS:

1. EXTRACT COMPLETE QUESTIONS:
   - Question numbers: "1.", "Q1", "Question 1", "(1)", "1)", etc.
   - Full question text including all details
   - All sub-parts: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
   - Mark allocations: [5 marks], [10], (3), etc.
   - Multiple choice options A, B, C, D with full text
   - Mathematical expressions, matrices, formulas
   - References to figures/diagrams

2. PRESERVE STRUCTURE:
   - Maintain question hierarchy
   - Keep proper indentation for sub-parts
   - Include all instructional text within questions
   - Preserve mathematical notation exactly

3. FORMAT REQUIREMENTS:
   - Start each main question: "Question [number]: [full question text]"
   - Sub-parts with proper indentation
   - Include mark allocations where present
   - For MCQs, include all options with full text
   - Note diagram references as [FIGURE/DIAGRAM REFERENCED]

4. IGNORE ADMINISTRATIVE CONTENT:
   - Headers, footers, institution names
   - Course codes, instructor names
   - Exam duration, total marks, date/time
   - General instructions not part of specific questions
   - Page numbers, watermarks

EXAMPLE OUTPUT FORMAT:
Question 1: Consider the following incidence matrix of a simple undirected graph. Convert this into an adjacency matrix representation. [2 marks]
Matrix:
1 0 0
1 1 1  
0 1 0
0 0 1

Question 2: Which network model assumes that edges are formed between pairs of nodes with a uniform probability, independent of other edges? [2 marks]
A. Barabási-Albert Model
B. Erdős–Rényi (Random Network) Model  
C. Watts-Strogatz (Small-World) Model
D. Configuration Model

Question 3: In game theory, a situation where no player can improve their outcome by unilaterally changing their strategy is known as: [2 marks]
(a) What is this concept called?
(b) Provide an example.

OUTPUT ONLY the extracted questions in this format. No explanations or markdown.'''
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            strategy = task["strategy"]
            
            # Convert PDF to images with higher DPI for better text recognition
            image_paths = pdf_to_images(file_path)
            
            # Choose model based on strategy
            model = strategy["recommended_model"]
            
            # Extract questions using chosen model
            question_text = self._extract_questions(image_paths, model)
            
            # Validate and enhance extraction
            validation = self._validate_extraction(question_text)
            
            # Retry with different model if validation fails
            if not validation["is_valid"] and validation["should_retry"]:
                fallback_model = "gemini" if model == "openai" else "openai"
                print(f"Retrying question extraction with {fallback_model}")
                question_text = self._extract_questions(image_paths, fallback_model)
                validation = self._validate_extraction(question_text)
                
                # If still failing, try enhanced extraction
                if not validation["is_valid"]:
                    question_text = self._enhanced_question_extraction(image_paths, model)
                    validation = self._validate_extraction(question_text)
            
            return AgentResult(
                success=validation["is_valid"],
                data={
                    "question_text": question_text,
                    "model_used": model,
                    "validation": validation,
                    "image_paths": image_paths
                },
                confidence=validation["confidence"]
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def _extract_questions(self, image_paths: List[str], model: str) -> str:
        if model == "gemini":
            return gemini_extract_question_text(image_paths, self.question_prompt)
        else:
            return gpt4o_extract_answer_latex(image_paths, question_text="", prompt=self.question_prompt)
    
    def _enhanced_question_extraction(self, image_paths: List[str], model: str) -> str:
        """Enhanced extraction with specific focus on question patterns"""
        enhanced_prompt = '''FOCUSED QUESTION EXTRACTION

Look for these specific patterns in the exam paper:

1. QUESTION NUMBERS: Look for "1.", "2.", "Question 1", "Q1", "(a)", "(b)", etc.
2. QUESTION INDICATORS: "Consider", "Which", "What", "How", "Explain", "Calculate", "Solve", "Find", "Describe"
3. MARK ALLOCATIONS: [2], [10 marks], (5 marks), etc.
4. MULTIPLE CHOICE: A., B., C., D. options
5. SUB-QUESTIONS: (a), (b), (c) or (i), (ii), (iii)

For each question found:
- Extract the complete question text
- Include all sub-parts
- Include mark allocations
- Include MCQ options if present
- Preserve mathematical expressions

Format as:
Question 1: [Full question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next question...]

Extract EVERYTHING that students need to answer. Be thorough and complete.'''
        
        if model == "gemini":
            return gemini_extract_question_text(image_paths, enhanced_prompt)
        else:
            return gpt4o_extract_answer_latex(image_paths, question_text="", prompt=enhanced_prompt)
    
    def _validate_extraction(self, question_text: str) -> Dict:
        validation = {
            "is_valid": True,
            "should_retry": False,
            "confidence": 0.9,
            "issues": []
        }
        
        if not question_text or len(question_text.strip()) < 100:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.1
            validation["issues"].append("Extracted text too short")
            return validation
        
        if "NO QUESTIONS FOUND" in question_text:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.0
            validation["issues"].append("No questions detected")
            return validation
        
        # Check for question patterns
        question_patterns = [
            r'Question\s+\d+',
            r'Q\d+',
            r'^\d+[\.:]\s',
            r'\(\d+\)',
            r'\d+\)',
            r'Consider',
            r'Which',
            r'What',
            r'How',
            r'Explain'
        ]
        
        import re
        has_questions = any(re.search(pattern, question_text, re.MULTILINE | re.IGNORECASE) 
                           for pattern in question_patterns)
        
        if not has_questions:
            validation["confidence"] *= 0.5
            validation["issues"].append("No clear question patterns found")
        
        # Check for mark allocations
        mark_patterns = [r'\[\d+\]', r'\[\d+\s*marks?\]', r'\(\d+\s*marks?\)']
        has_marks = any(re.search(pattern, question_text, re.IGNORECASE) 
                       for pattern in mark_patterns)
        
        if has_marks:
            validation["confidence"] = min(validation["confidence"] + 0.1, 1.0)
        
        # Check for multiple choice patterns
        mcq_patterns = [r'A\.\s', r'B\.\s', r'C\.\s', r'D\.\s']
        has_mcq = any(re.search(pattern, question_text) for pattern in mcq_patterns)
        
        if has_mcq:
            validation["confidence"] = min(validation["confidence"] + 0.1, 1.0)
        
        return validation