# agents/question_extractor.py
from typing import Dict, List, Any
from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex
from utils.ocr_gemini import gemini_extract_question_text
from .base_agent import BaseAgent, AgentResult

class QuestionExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__("QuestionExtractor", ["openai_vision", "gemini_vision"])
        self.question_prompt = '''CRITICAL QUESTION PAPER EXTRACTION TASK:

You are extracting questions from an academic examination paper. Your task is to identify and transcribe ONLY the actual questions that require answers from students.

STRICT REQUIREMENTS:

1. IDENTIFY ACTUAL QUESTIONS - Look for content that:
   - Has clear question numbers (1, 2, 3... or Q1, Q2, Q3... or Question 1, Question 2...)
   - Contains question text that asks something requiring an answer
   - May have mark allocation like [5], [10], (3 marks), etc.
   - May have sub-parts (a), (b), (c)... or i), ii), iii... or (1), (2), (3)...
   - May contain multiple choice options A, B, C, D
   - May reference figures, diagrams, matrices, or mathematical expressions

2. EXCLUDE THE FOLLOWING - Do NOT include:
   - Header information (Department, Institute name, Course codes, Instructor names)
   - Exam metadata (Date, Time, Duration, Total marks)
   - General instructions ("The exam is closed book", "Write clearly", "Do not use AI")
   - Page numbers, watermarks, or footer text
   - Phrases like "Please go on to the next page", "Question continues on next page"
   - "End of Exam" or similar closing statements

3. FORMATTING REQUIREMENTS:
   - Start each main question with "Question [number]:"
   - Preserve all sub-parts with proper indentation
   - Include mark allocations if present: [X marks] or [X]
   - For multiple choice questions, include all options A, B, C, D
   - Preserve mathematical expressions, matrices, and formulas
   - Note references to figures/diagrams as [FIGURE REFERENCED] or [DIAGRAM REFERENCED]
   - Maintain question hierarchy and structure

If no actual questions are found, state "NO QUESTIONS FOUND ON THIS PAGE"

OUTPUT ONLY the extracted question text without any explanations or markdown formatting.'''
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            strategy = task["strategy"]
            
            # Convert PDF to images
            image_paths = pdf_to_images(file_path)
            
            # Choose model based on strategy
            model = strategy["recommended_model"]
            
            # Extract questions using chosen model
            question_text = self._extract_questions(image_paths, model)
            
            # Validate extraction
            validation = self._validate_extraction(question_text)
            
            # Retry with different model if validation fails
            if not validation["is_valid"] and validation["should_retry"]:
                fallback_model = "gemini" if model == "openai" else "openai"
                question_text = self._extract_questions(image_paths, fallback_model)
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
    
    def _validate_extraction(self, question_text: str) -> Dict:
        validation = {
            "is_valid": True,
            "should_retry": False,
            "confidence": 0.9,
            "issues": []
        }
        
        if not question_text or len(question_text.strip()) < 50:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.1
            validation["issues"].append("Extracted text too short")
        
        if "NO QUESTIONS FOUND" in question_text:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.0
            validation["issues"].append("No questions detected")
        
        # Check for question patterns
        question_patterns = ["Question 1", "1.", "Q1", "question 1"]
        has_questions = any(pattern.lower() in question_text.lower() for pattern in question_patterns)
        
        if not has_questions:
            validation["confidence"] *= 0.5
            validation["issues"].append("No clear question patterns found")
        
        return validation