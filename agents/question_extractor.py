# agents/question_extractor.py - Enhanced multi-page support
from typing import Dict, List, Any
from utils.ocr_openai import pdf_to_images, gpt4o_extract_questions
from utils.ocr_gemini import gemini_extract_question_text
from .base_agent import BaseAgent, AgentResult

class QuestionExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__("QuestionExtractor", ["openai_vision", "gemini_vision"])
        self.question_prompt = '''📝 EXTRACT QUESTIONS ONLY FROM EXAM PAPER - NOT ANSWERS

🚨 CRITICAL INSTRUCTION: You are viewing a QUESTION PAPER (printed exam). Extract ONLY the questions that students need to answer, NOT any solutions, answers, or student work.

📋 MANDATORY EXTRACTION RULES:

1. ✅ EXTRACT QUESTIONS ONLY:
   - Question numbers: "Question 1:", "Q1", "1.", "(a)", "(b)", etc.
   - Complete question text and instructions
   - Multiple choice options (A, B, C, D)
   - Mathematical expressions and formulas
   - Mark allocations: [2], [10 marks], etc.
   - Figure/diagram references

2. ❌ DO NOT EXTRACT:
   - Any text labeled "Solution:", "Answer:", "Student response"
   - Handwritten content (this is a printed question paper)
   - Sample answers or model solutions
   - Grading rubrics or marking schemes
   - Any content that looks like student work

3. 📖 QUESTION PAPER IDENTIFICATION:
   - Focus on printed/typed text (the official questions)
   - Ignore any handwritten annotations or solutions
   - Extract what students are supposed to answer
   - Include complete question statements

4. 📄 MULTI-PAGE PROCESSING:
   - Process ALL pages of the question paper
   - Maintain question sequence across all pages
   - Continue from page 1 through final page
   - Extract every question completely
   - Continue extracting beyond (f) to (g), (h), (i), (j), (k)...
   - Extract questions beyond 5 to 6, 7, 8, 9, 10...

5. 📝 OUTPUT FORMAT:
Question 1: [Complete question text] [marks]
(a) [Sub-question text]
(b) [Sub-question text]
(c) [Sub-question text]
...continue to (j), (k), etc. if they exist

Question 2: [Complete question text] [marks]
A. [Option A text]
B. [Option B text]
C. [Option C text]
D. [Option D text]

Question 3: [Complete question text] [marks]
(i) [Sub-question text]
(ii) [Sub-question text]
...continue to (x), (xx), etc. if they exist

[Continue for ALL questions across ALL pages - don't stop early]

🎯 GOAL: Extract complete question paper so students know what to answer.

EXTRACT ONLY QUESTIONS - NOT ANSWERS OR SOLUTIONS.'''
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            strategy = task["strategy"]
            
            print(f"Extracting questions from: {file_path}")
            
            # Convert PDF to images with higher DPI for better text recognition
            image_paths = pdf_to_images(file_path)
            print(f"Processing {len(image_paths)} pages for question extraction")
            
            # Choose model based on strategy
            model = strategy["recommended_model"]
            print(f"Using {model.upper()} for question extraction")
            
            # Extract questions using chosen model
            question_text = self._extract_questions_multipage(image_paths, model)
            print(f"Extracted {len(question_text)} characters from {len(image_paths)} pages")
            
            # Validate and enhance extraction
            validation = self._validate_multipage_extraction(question_text, len(image_paths))
            print(f"Validation result: {validation['confidence']:.2f} confidence, valid: {validation['is_valid']}")
            
            # Retry with different model if validation fails
            if not validation["is_valid"] and validation["should_retry"]:
                fallback_model = "gemini" if model == "openai" else "openai"
                print(f"Retrying question extraction with {fallback_model.upper()}")
                question_text = self._extract_questions_multipage(image_paths, fallback_model)
                validation = self._validate_multipage_extraction(question_text, len(image_paths))
                
                # If still failing, try enhanced extraction
                if not validation["is_valid"]:
                    print("Trying enhanced page-by-page extraction...")
                    question_text = self._enhanced_question_extraction_multipage(image_paths, model)
                    validation = self._validate_multipage_extraction(question_text, len(image_paths))
                    
                    # Final attempt with combined approach if still failing
                    if not validation["is_valid"] and len(image_paths) > 1:
                        print("Trying combined sequential + parallel extraction...")
                        question_text = self._combined_extraction_approach(image_paths, model)
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
            print(f"Error in question extraction: {e}")
            return AgentResult(success=False, error=str(e))
    
    def _extract_questions_multipage(self, image_paths: List[str], model: str) -> str:
        """Extract questions with multi-page awareness"""
        if model == "gemini":
            return gemini_extract_question_text(image_paths, self.question_prompt)
        else:
            return gpt4o_extract_questions(image_paths, self.question_prompt)
    
    def _enhanced_question_extraction_multipage(self, image_paths: List[str], model: str) -> str:
        """Enhanced extraction with page-by-page processing for difficult cases"""
        print(f"Enhanced multi-page extraction with {model.upper()}")
        
        enhanced_prompt = '''🚨 EMERGENCY COMPLETE EXTRACTION PROTOCOL 🚨

CRITICAL MISSION: Extract EVERY SINGLE QUESTION from the ENTIRE examination paper.

⚠️ EXTRACTION REQUIREMENTS - NO EXCEPTIONS:

1. 🔍 SCAN EVERY MILLIMETER OF ALL PAGES:
   - Search for questions 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15... (unlimited)
   - Find sub-parts (a), (b), (c), (d), (e), (f), (g), (h), (i), (j), (k), (l)... (complete alphabet)
   - Look for roman numerals (i), (ii), (iii), (iv), (v), (vi), (vii)... (all that exist)
   - Check for numbered sub-parts (1), (2), (3), (4), (5)... (all that exist)

2. 🎯 EXHAUSTIVE PATTERN DETECTION:
   - "Question X:" where X = 1,2,3,4,5,6,7,8,9,10+ (all numbers)
   - "QX" patterns (Q1, Q2, Q3... Q20, Q25, Q30...)
   - "X." patterns (1., 2., 3... 50. if they exist)
   - MCQ patterns: A, B, C, D, E, F... (all options)
   - Math expressions, matrices, diagrams

3. 📝 COMPLETE OUTPUT FORMAT:
Question 1: [COMPLETE text with ALL details] [marks]
(a) [Complete sub-question]
(b) [Complete sub-question]
(c) [Complete sub-question]
(d) [Complete sub-question]
(e) [Complete sub-question]
(f) [Complete sub-question]
[Continue through ALL sub-parts - don't stop at (f)]

Question 2: [COMPLETE text] [marks]
A. [Complete option]
B. [Complete option]
C. [Complete option]
D. [Complete option]
[Continue through ALL options]

[CONTINUE FOR ALL QUESTIONS - DON'T STOP EARLY]

🚨 MISSION CRITICAL POINTS:
- Extract sub-parts beyond (f) - continue to (g), (h), (i), (j), (k)...
- Extract questions beyond 5 - continue to 6, 7, 8, 9, 10, 11...
- Process ALL pages thoroughly
- Include ALL mathematical expressions
- Include ALL detailed descriptions

⛔ PROHIBITED ACTIONS:
- Stopping at question (f)
- Stopping at question 5
- Summarizing content
- Skipping any pages
- Truncating text

EXTRACT ABSOLUTELY EVERYTHING FROM THE COMPLETE DOCUMENT.'''
        
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
        
        # Dynamic minimum length based on page count
        min_total_length = max(50, num_pages * 30)  # At least 50 chars, or 30 per page
        if not question_text or len(question_text.strip()) < min_total_length:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.1
            validation["issues"].append(f"Extracted text too short (got {len(question_text)} chars, need {min_total_length})")
            return validation
        
        if "NO QUESTIONS FOUND" in question_text.upper():
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.0
            validation["issues"].append("No questions detected")
            return validation
        
        # Enhanced multi-page specific validation
        if num_pages > 1:
            expected_min_length = num_pages * 80  # Reduced to 80 chars per page for more realistic validation
            if len(question_text) < expected_min_length:
                validation["confidence"] *= 0.7
                validation["issues"].append(f"Content seems short for {num_pages} pages")
        
        # Check for question patterns
        import re
        question_patterns = [
            r'Question\s+\d+',
            r'Q\.\s*\d+',
            r'Q\d+',
            r'^\d+[\.:]\s',
            r'\(\d+\)',
            r'\d+\)',
            r'\(\w\)',  # (a), (b), (c) patterns
            r'Consider',
            r'Which',
            r'What',
            r'How',
            r'Explain',
            r'Calculate',
            r'Find',
            r'Solve',
            r'Show',
            r'Prove'
        ]
        
        question_matches = []
        for pattern in question_patterns:
            matches = re.findall(pattern, question_text, re.MULTILINE | re.IGNORECASE)
            question_matches.extend(matches)
        
        if not question_matches:
            validation["confidence"] *= 0.3
            validation["should_retry"] = True
            validation["issues"].append("No clear question patterns found")
        
        # Count actual questions using multiple patterns
        question_count_patterns = [
            r'Question\s+\d+',
            r'Q\.\s*\d+',
            r'Q\d+',
            r'^\d+[\.:]\s'
        ]
        
        question_counts = []
        for pattern in question_count_patterns:
            matches = re.findall(pattern, question_text, re.MULTILINE | re.IGNORECASE)
            question_counts.append(len(matches))
        
        # Use the highest count found
        question_count = max(question_counts) if question_counts else 0
        
        # Also check for sub-question patterns that might indicate more questions
        sub_question_patterns = [r'\([a-z]\)', r'\([ivx]+\)', r'\(\d+\)']
        sub_question_count = 0
        for pattern in sub_question_patterns:
            sub_question_count += len(re.findall(pattern, question_text, re.IGNORECASE))
        
        # If we have many sub-questions, there might be more main questions
        if sub_question_count > 10 and question_count < 3:
            question_count = max(question_count, sub_question_count // 6)  # Estimate main questions
        
        # More lenient question count validation for multi-page
        if num_pages > 2 and question_count < 1:
            validation["confidence"] *= 0.7
            validation["issues"].append(f"Only {question_count} questions found across {num_pages} pages")
        elif num_pages > 4 and question_count < 2:
            validation["confidence"] *= 0.8
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
        
        print(f"Validation Details:")
        print(f"   • {question_count} questions found")
        print(f"   • {len(question_text)} total characters")
        print(f"   • {content_per_page:.0f} characters per page")
        print(f"   • Mark allocations: {'Yes' if has_marks else 'No'}")
        print(f"   • MCQ options: {mcq_count}")
        print(f"   • Issues: {validation['issues'] if validation['issues'] else 'None'}")
        
        return validation
    
    def _combined_extraction_approach(self, image_paths: List[str], model: str) -> str:
        """Combined approach: Extract each page individually then merge intelligently"""
        print(f"Using combined extraction approach for {len(image_paths)} pages")
        
        page_extractions = []
        question_number_tracker = 1
        
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            print(f"Processing page {page_num}/{len(image_paths)}")
            
            # Create page-specific prompt
            page_prompt = f'''EXTRACT QUESTIONS FROM PAGE {page_num} OF {len(image_paths)}

This is page {page_num} of a {len(image_paths)}-page examination paper.

CRITICAL INSTRUCTIONS:
1. Extract ALL questions visible on THIS page
2. Continue question numbering from previous pages (start from Question {question_number_tracker} if this is the first question on this page)
3. Include complete question text with all details
4. Include all sub-parts: (a), (b), (c), etc.
5. Include mark allocations: [marks]
6. Include MCQ options if present: A., B., C., D.
7. Preserve mathematical expressions exactly

LOOK FOR:
- Question numbers: "Question X", "Q.X", "X.", "(X)"
- Continuation of questions from previous pages
- New questions starting on this page
- Sub-questions and parts

FORMAT:
Question X: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

For MCQs:
A. [Complete option text]
B. [Complete option text]
C. [Complete option text]
D. [Complete option text]

Extract ALL question content from this page. Include everything students need to answer.'''
            
            # Extract from this page
            if model == "gemini":
                from utils.ocr_gemini import gemini_extract_question_text
                # Process single page with rate limiting handled internally
                page_result = gemini_extract_question_text([image_path], page_prompt)
            else:
                from utils.ocr_openai import gpt4o_extract_questions
                # Process single page (no rate limiting needed for OpenAI)
                page_result = gpt4o_extract_questions([image_path], page_prompt)
            
            # Clean and validate page result
            if page_result and len(page_result.strip()) > 50:
                # Update question number tracker based on what we found
                import re
                questions_found = re.findall(r'Question\s+(\d+)', page_result, re.IGNORECASE)
                if questions_found:
                    max_question_num = max(int(q) for q in questions_found)
                    question_number_tracker = max_question_num + 1
                
                page_extractions.append({
                    'page': page_num,
                    'content': page_result.strip(),
                    'questions_found': len(questions_found)
                })
                print(f"Page {page_num}: extracted {len(page_result)} chars, {len(questions_found)} questions")
            else:
                print(f"Page {page_num}: minimal content extracted")
        
        # Combine all page extractions intelligently
        if not page_extractions:
            return "No questions extracted from any page"
        
        combined_content = []
        combined_content.append("=== MULTI-PAGE QUESTION EXTRACTION ===")
        combined_content.append(f"Processed {len(image_paths)} pages")
        combined_content.append("")
        
        for extraction in page_extractions:
            combined_content.append(f"=== PAGE {extraction['page']} ===")
            combined_content.append(extraction['content'])
            combined_content.append("")
        
        final_result = "\n".join(combined_content)
        print(f"Combined extraction: {len(final_result)} total characters from {len(page_extractions)} pages")
        
        return final_result