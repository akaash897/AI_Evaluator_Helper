# agents/question_extractor.py - Enhanced multi-page support
from typing import Dict, List, Any
from utils.ocr_openai import pdf_to_images, gpt4o_extract_questions
from utils.ocr_gemini import gemini_extract_question_text
from .base_agent import BaseAgent, AgentResult

class QuestionExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__("QuestionExtractor", ["openai_vision", "gemini_vision"])
        self.question_prompt = '''EXTRACT ALL EXAMINATION QUESTIONS FROM ALL PAGES

CRITICAL: This is a MULTI-PAGE examination paper. You must extract questions from EVERY page provided.

You are analyzing a complete academic examination paper. Extract ALL questions from ALL pages.

MULTI-PAGE REQUIREMENTS:

1. PROCESS ALL PAGES: Look through every image/page provided - skip none
2. EXTRACT FROM EACH PAGE: Find questions on every page, not just the first
3. MAINTAIN ORDER: Extract questions in sequence across all pages
4. COMPLETE DOCUMENT: This exam may span multiple pages - process everything

5. IDENTIFY ALL QUESTIONS ACROSS ALL PAGES:
   - Question numbers: "1.", "Q1", "Question 1", "(1)", "1)", etc.
   - Full question text including all details
   - All sub-parts: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
   - Mark allocations: [5 marks], [10], (3), etc.
   - Multiple choice options A, B, C, D with full text
   - Mathematical expressions, matrices, formulas
   - References to figures/diagrams

6. PRESERVE STRUCTURE ACROSS PAGES:
   - Maintain question hierarchy
   - Keep proper indentation for sub-parts
   - Include all instructional text within questions
   - Preserve mathematical notation exactly

7. FORMAT REQUIREMENTS FOR ALL PAGES:
   - Start each main question: "Question [number]: [full question text]"
   - Sub-parts with proper indentation
   - Include mark allocations where present
   - For MCQs, include all options with full text
   - Note diagram references as [FIGURE/DIAGRAM REFERENCED]

8. IGNORE ADMINISTRATIVE CONTENT ON ALL PAGES:
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

OUTPUT ONLY the extracted questions from ALL pages in this format. No explanations or markdown.'''
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            strategy = task["strategy"]
            
            print(f"🔍 Extracting questions from: {file_path}")
            
            # Convert PDF to images with higher DPI for better text recognition
            image_paths = pdf_to_images(file_path)
            print(f"📄 Processing {len(image_paths)} pages for question extraction")
            
            # Choose model based on strategy
            model = strategy["recommended_model"]
            print(f"🤖 Using {model.upper()} for question extraction")
            
            # Extract questions using chosen model
            question_text = self._extract_questions_multipage(image_paths, model)
            print(f"📝 Extracted {len(question_text)} characters from {len(image_paths)} pages")
            
            # Validate and enhance extraction
            validation = self._validate_multipage_extraction(question_text, len(image_paths))
            print(f"✅ Validation result: {validation['confidence']:.2f} confidence, valid: {validation['is_valid']}")
            
            # Retry with different model if validation fails
            if not validation["is_valid"] and validation["should_retry"]:
                fallback_model = "gemini" if model == "openai" else "openai"
                print(f"🔄 Retrying question extraction with {fallback_model.upper()}")
                question_text = self._extract_questions_multipage(image_paths, fallback_model)
                validation = self._validate_multipage_extraction(question_text, len(image_paths))
                
                # If still failing, try enhanced extraction
                if not validation["is_valid"]:
                    print("🔧 Trying enhanced page-by-page extraction...")
                    question_text = self._enhanced_question_extraction_multipage(image_paths, model)
                    validation = self._validate_multipage_extraction(question_text, len(image_paths))
            
            return AgentResult(
                success=validation["is_valid"],
                data={
                    "question_text": question_text,
                    "model_used": model,
                    "validation": validation,
                    "image_paths": image_paths,
                    "pages_processed": len(image_paths)
                },
                confidence=validation["confidence"]
            )
            
        except Exception as e:
            print(f"❌ Error in question extraction: {e}")
            return AgentResult(success=False, error=str(e))
    
    def _extract_questions_multipage(self, image_paths: List[str], model: str) -> str:
        """Extract questions with multi-page awareness"""
        if model == "gemini":
            return gemini_extract_question_text(image_paths, self.question_prompt)
        else:
            return gpt4o_extract_questions(image_paths, self.question_prompt)
    
    def _enhanced_question_extraction_multipage(self, image_paths: List[str], model: str) -> str:
        """Enhanced extraction with page-by-page processing for difficult cases"""
        print(f"🔍 Enhanced multi-page extraction with {model.upper()}")
        
        enhanced_prompt = '''FOCUSED MULTI-PAGE QUESTION EXTRACTION

CRITICAL: Process ALL pages of this examination paper.

Look for these specific patterns across ALL pages:

1. QUESTION NUMBERS ON ALL PAGES: "1.", "2.", "Question 1", "Q1", "(a)", "(b)", etc.
2. QUESTION INDICATORS: "Consider", "Which", "What", "How", "Explain", "Calculate", "Solve", "Find", "Describe"
3. MARK ALLOCATIONS: [2], [10 marks], (5 marks), etc.
4. MULTIPLE CHOICE: A., B., C., D. options
5. SUB-QUESTIONS: (a), (b), (c) or (i), (ii), (iii)

For each question found on any page:
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

Extract EVERYTHING students need to answer from ALL pages. Be thorough and complete across the entire document.'''
        
        if model == "gemini":
            return gemini_extract_question_text(image_paths, enhanced_prompt)
        else:
            return gpt4o_extract_questions(image_paths, enhanced_prompt)
    
    def _validate_multipage_extraction(self, question_text: str, num_pages: int) -> Dict:
        """Enhanced validation for multi-page extraction"""
        validation = {
            "is_valid": True,
            "should_retry": False,
            "confidence": 0.9,
            "issues": [],
            "pages_processed": num_pages
        }
        
        if not question_text or len(question_text.strip()) < 100:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.1
            validation["issues"].append("Extracted text too short")
            return validation
        
        if "NO QUESTIONS FOUND" in question_text.upper():
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.0
            validation["issues"].append("No questions detected")
            return validation
        
        # Enhanced multi-page specific validation
        if num_pages > 1:
            expected_min_length = num_pages * 200  # At least 200 chars per page
            if len(question_text) < expected_min_length:
                validation["confidence"] *= 0.6
                validation["issues"].append(f"Content seems short for {num_pages} pages")
        
        # Check for question patterns
        import re
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
        
        question_matches = []
        for pattern in question_patterns:
            matches = re.findall(pattern, question_text, re.MULTILINE | re.IGNORECASE)
            question_matches.extend(matches)
        
        if not question_matches:
            validation["confidence"] *= 0.3
            validation["should_retry"] = True
            validation["issues"].append("No clear question patterns found")
        
        # Count actual questions
        question_count = len(re.findall(r'Question\s+\d+', question_text, re.IGNORECASE))
        
        if num_pages > 1 and question_count < 2:
            validation["confidence"] *= 0.7
            validation["issues"].append(f"Only {question_count} questions found across {num_pages} pages")
        
        # Check for mark allocations (good indicator of real questions)
        mark_patterns = [r'\[\d+\]', r'\[\d+\s*marks?\]', r'\(\d+\s*marks?\)']
        has_marks = any(re.search(pattern, question_text, re.IGNORECASE) 
                       for pattern in mark_patterns)
        
        if has_marks:
            validation["confidence"] = min(validation["confidence"] + 0.1, 1.0)
        else:
            validation["confidence"] *= 0.8
            validation["issues"].append("No mark allocations found")
        
        # Check for multiple choice patterns
        mcq_patterns = [r'A\.\s', r'B\.\s', r'C\.\s', r'D\.\s']
        mcq_count = sum(len(re.findall(pattern, question_text)) for pattern in mcq_patterns)
        
        if mcq_count >= 4:  # At least one complete MCQ
            validation["confidence"] = min(validation["confidence"] + 0.1, 1.0)
        
        # Check for page indicators (if extraction was page-by-page)
        if "PAGE" in question_text.upper() and num_pages > 1:
            validation["confidence"] = min(validation["confidence"] + 0.05, 1.0)
            validation["issues"].append("Page-by-page extraction detected")
        
        # Final confidence adjustment based on content length and pages
        content_per_page = len(question_text) / num_pages if num_pages > 0 else len(question_text)
        if content_per_page > 500:  # Good amount of content per page
            validation["confidence"] = min(validation["confidence"] + 0.1, 1.0)
        
        print(f"📊 Validation Details:")
        print(f"   • {question_count} questions found")
        print(f"   • {len(question_text)} total characters")
        print(f"   • {content_per_page:.0f} characters per page")
        print(f"   • Mark allocations: {'Yes' if has_marks else 'No'}")
        print(f"   • MCQ options: {mcq_count}")
        print(f"   • Issues: {validation['issues'] if validation['issues'] else 'None'}")
        
        return validation