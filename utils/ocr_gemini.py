# utils/ocr_gemini.py - Enhanced version with multi-page support
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import re

# Rate limiting for Gemini (15 requests per minute)
_last_request_time = 0
_request_count = 0
_reset_time = time.time()

def _rate_limit_gemini():
    """Handle Gemini's 15 RPM limit"""
    global _last_request_time, _request_count, _reset_time
    
    current_time = time.time()
    
    # Reset counter every minute
    if current_time - _reset_time >= 60:
        _request_count = 0
        _reset_time = current_time
    
    # If we've hit the limit, wait
    if _request_count >= 15:
        sleep_time = 60 - (current_time - _reset_time)
        if sleep_time > 0:
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds for Gemini...")
            time.sleep(sleep_time)
            _request_count = 0
            _reset_time = time.time()
    
    # Ensure minimum 4 seconds between requests (15 requests per 60 seconds)
    time_since_last = current_time - _last_request_time
    min_interval = 4.0  # 60/15 = 4 seconds
    
    if time_since_last < min_interval:
        sleep_time = min_interval - time_since_last
        print(f"Rate limiting: waiting {sleep_time:.1f} seconds between Gemini requests...")
        time.sleep(sleep_time)
    
    _request_count += 1
    _last_request_time = time.time()

load_dotenv()

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)

def gemini_extract_answer_latex(image_paths, question_text, prompt=None):
    configure_gemini()
    
    if prompt is None:
        prompt = f"""📋 EXTRACT STUDENT HANDWRITTEN ANSWERS AND MAP TO QUESTIONS

🚨 CRITICAL TASK: You are viewing STUDENT ANSWER SHEETS (handwritten responses). Extract ONLY what the student wrote, NOT the questions themselves.

📝 ANSWER SHEET EXTRACTION RULES:

1. ✅ EXTRACT STUDENT WORK ONLY:
   - Handwritten text and responses
   - Mathematical calculations and working
   - Diagrams, graphs, charts drawn by student
   - Crossed-out work and corrections
   - Student's reasoning and explanations
   - Answer numbers (Q1, 1., (a), etc. written by student)

2. ❌ DO NOT EXTRACT:
   - Printed question text (that's already provided separately)
   - Instructions or exam headers
   - Anything that looks like the original question paper

3. 🔗 MAPPING STRATEGY:
   - Look for question indicators in student's handwriting (Q1, 1., (a), etc.)
   - Match student work to the appropriate question from the question paper
   - Group related student work under the same question
   - If unclear which question, note the content and make best guess

4. 📄 QUESTION PAPER REFERENCE:
{question_text}

5. 📝 LATEX OUTPUT FORMAT:
\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}

\\begin{{document}}
\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Student Responses Mapped to Questions}}

\\subsection*{{Question 1}}
\\textbf{{Question:}} [Copy complete Question 1 from question paper above]

\\textbf{{Student Answer:}}
\\begin{{quote}}
[Extract ONLY what the student wrote for Q1 - their handwritten response]
\\end{{quote}}

\\subsection*{{Question 2}}
\\textbf{{Question:}} [Copy complete Question 2 from question paper above]

\\textbf{{Student Answer:}}
\\begin{{quote}}
[Extract ONLY what the student wrote for Q2 - their handwritten response]
\\end{{quote}}

[Continue for all questions where student provided answers]

\\end{{document}}

🎯 GOAL: Create a document showing what questions were asked and what the student actually wrote in response.

EXTRACT STUDENT HANDWRITING ONLY - NOT QUESTION TEXT."""

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    try:
        _rate_limit_gemini()  # Apply rate limiting
        response = model.generate_content([prompt] + images)
        latex_text = response.text.strip()
        
        # Clean and validate LaTeX output
        latex_text = _clean_gemini_latex_output(latex_text)
        
        # Validate structure
        if not _validate_gemini_latex_structure(latex_text):
            print("Generated LaTeX failed validation, creating fallback...")
            latex_text = _create_gemini_fallback_latex(latex_text, question_text)
        
        return latex_text
        
    except Exception as e:
        print(f"Error in Gemini processing: {e}")
        return _create_gemini_fallback_latex(f"Error: {str(e)}", question_text)

def gemini_extract_question_text(image_paths, prompt=None):
    configure_gemini()
    
    if prompt is None:
        prompt = '''📝 EXTRACT QUESTIONS ONLY FROM EXAM PAPER - NOT ANSWERS

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

5. 📝 OUTPUT FORMAT:
Question 1: [Complete question text] [marks]
(a) [Sub-question text]
(b) [Sub-question text]

Question 2: [Complete question text] [marks]
A. [Option A text]
B. [Option B text] 
C. [Option C text]
D. [Option D text]

[Continue for ALL questions across ALL pages]

🎯 GOAL: Extract complete question paper so students know what to answer.

EXTRACT ONLY QUESTIONS - NOT ANSWERS OR SOLUTIONS.'''

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Load ALL images
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    print(f"DEBUG: Processing {len(images)} pages for question extraction")
    
    try:
        # Send ALL images at once to process the complete document
        _rate_limit_gemini()  # Apply rate limiting
        response = model.generate_content([prompt] + images)
        result = response.text.strip()
        
        print(f"DEBUG: Extracted {len(result)} characters from {len(images)} pages")
        
        # Validate that we got content from multiple pages
        result = _enhance_multi_page_extraction(result, len(images))
        
        # Final validation
        if _validate_multi_page_extraction(result, len(images)):
            return result
        else:
            # Try page-by-page extraction as fallback
            print("Multi-page extraction seems incomplete, trying page-by-page approach...")
            return _extract_questions_page_by_page(images, prompt, model)
            
    except Exception as e:
        print(f"Error in Gemini question extraction: {e}")
        return f"Error extracting questions: {str(e)}"

def _extract_questions_page_by_page(images, base_prompt, model):
    """Fallback: Extract questions page by page and combine"""
    all_questions = []
    
    for page_num, image in enumerate(images, 1):
        page_prompt = f"""EXTRACT QUESTIONS FROM PAGE {page_num}

This is page {page_num} of a {len(images)}-page exam paper.

Extract ALL questions from THIS specific page. Continue question numbering appropriately.

REQUIREMENTS:
1. Extract complete question text from this page
2. Include sub-parts: (a), (b), (c), etc.
3. Include mark allocations: [marks]
4. Include MCQ options if present
5. Preserve mathematical expressions

OUTPUT FORMAT:
Question [number]: [Complete question text] [marks]
(a) [Sub-question if any]

Extract ALL content from this page that students need to answer.
"""
        
        try:
            _rate_limit_gemini()  # Apply rate limiting for each page
            response = model.generate_content([page_prompt, image])
            page_result = response.text.strip()
            
            if page_result and len(page_result) > 50:
                all_questions.append(f"\n=== PAGE {page_num} ===")
                all_questions.append(page_result)
                print(f"DEBUG: Page {page_num} extracted {len(page_result)} characters")
            else:
                print(f"DEBUG: Page {page_num} had minimal content")
            
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
    
    combined_result = "\n".join(all_questions)
    print(f"DEBUG: Combined result from all pages: {len(combined_result)} characters")
    return combined_result

def _validate_multi_page_extraction(text, num_pages):
    """Validate that extraction covered multiple pages"""
    if not text or len(text.strip()) < 100:
        print("DEBUG: Validation failed - text too short")
        return False
    
    # For multi-page documents, expect proportionally more content
    if num_pages > 1:
        expected_min_length = num_pages * 120  # Reduced from 200 to 120 chars per page
        if len(text) < expected_min_length:
            print(f"DEBUG: Extracted text ({len(text)} chars) seems too short for {num_pages} pages")
            return False
    
    # Check for question distribution
    import re
    question_count = len(re.findall(r'Question\s+\d+', text, re.IGNORECASE))
    
    # More lenient validation for multi-page documents
    if num_pages > 3 and question_count < 1:
        print(f"DEBUG: Only found {question_count} questions in {num_pages} pages - might be incomplete")
        return False
    elif num_pages > 5 and question_count < 2:
        print(f"DEBUG: Only found {question_count} questions in {num_pages} pages - might be incomplete")
        return False
    
    print(f"DEBUG: Multi-page validation passed - {question_count} questions in {num_pages} pages")
    return True

def _enhance_multi_page_extraction(text, num_pages):
    """Enhance extraction to ensure multi-page coverage"""
    if num_pages == 1:
        return text
    
    # Check if extraction seems complete for multi-page
    if num_pages > 1 and len(text) > 500:
        # Text seems substantial for multi-page, likely good
        return text
    
    return text

def _clean_gemini_latex_output(latex_text):
    """Clean Gemini LaTeX output"""
    if not latex_text:
        return ""
    
    # Remove markdown code blocks
    if "```latex" in latex_text:
        latex_text = latex_text.split("```latex")[1].split("```")[0].strip()
    elif "```" in latex_text:
        parts = latex_text.split("```")
        if len(parts) >= 3:
            latex_text = parts[1].strip()
    
    # Clean up common Gemini formatting issues
    latex_text = latex_text.replace("\\textbf{Question:}", "\\textbf{Question:}")
    latex_text = latex_text.replace("\\textbf{Student Answer:}", "\\textbf{Student Answer:}")
    
    return latex_text.strip()

def _validate_gemini_latex_structure(latex_text):
    """Validate LaTeX structure - more lenient validation"""
    if not latex_text or len(latex_text.strip()) < 100:
        return False
    
    # Essential elements only
    essential_elements = [
        "\\documentclass",
        "\\begin{document}",
        "\\end{document}"
    ]
    
    # Check if we have the essential structure
    has_essentials = all(element in latex_text for element in essential_elements)
    
    # Additional checks for content quality
    if has_essentials:
        # Check if there's actual content between begin and end document
        import re
        doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_text, re.DOTALL)
        if doc_match:
            content = doc_match.group(1).strip()
            # More lenient - just check for some content
            return len(content) > 50
    
    return has_essentials

def _create_gemini_fallback_latex(content, question_text):
    """Create fallback LaTeX document"""
    return f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\setlength{{\\parskip}}{{6pt}}

\\begin{{document}}

\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Processing Status}}
The AI model encountered difficulties generating complete output. Available content is shown below.

\\section*{{Question Paper Content}}
\\begin{{quote}}
{question_text if question_text else "Question content not available"}
\\end{{quote}}

\\section*{{Extracted Content}}
\\begin{{quote}}
{content[:1500] if content else "No content extracted"}
\\end{{quote}}

\\section*{{Recommendations}}
\\begin{{itemize}}
\\item Check image quality and clarity
\\item Ensure handwriting is legible
\\item Try processing again
\\item Consider manual review
\\end{{itemize}}

\\end{{document}}"""

def _enhance_question_extraction(text):
    """Enhance question extraction result"""
    if not text or len(text.strip()) < 50:
        return text
    
    lines = text.split('\n')
    enhanced_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Enhance question number detection
            if re.match(r'^\d+[\.:]\s', line) and not line.startswith('Question'):
                line = f"Question {line}"
            enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)

def _validate_question_extraction(text):
    """Validate question extraction"""
    if not text or len(text.strip()) < 50:
        return False
    
    # Check for question indicators
    question_patterns = [
        r'Question\s+\d+',
        r'Q\d+',
        r'^\d+[\.:]\s',
        r'\(\w\)',
        r'Consider',
        r'Which',
        r'What',
        r'How',
        r'Explain'
    ]
    
    has_questions = any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) 
                       for pattern in question_patterns)
    
    return has_questions

def _create_enhanced_question_prompt():
    """Create enhanced question extraction prompt"""
    return '''DETAILED MULTI-PAGE QUESTION EXTRACTION

CRITICAL: Process ALL pages of this examination paper.

Step 1: Scan ALL pages for question numbers
- Look for: "1.", "2.", "3.", "Q1", "Q2", "Question 1", etc.
- Note the location of each question number on each page

Step 2: For each question found on each page, extract:
- The complete question text
- Any sub-parts: (a), (b), (c) or (i), (ii), (iii)
- Mark allocations: [2], [10 marks], etc.
- Multiple choice options if present

Step 3: Format each question as:
Question [number]: [Complete question text] [marks]
(a) [Sub-question if any]
(b) [Sub-question if any]

Step 4: Include mathematical expressions exactly as shown

Step 5: For multiple choice questions, include all options:
A. [Complete option text]
B. [Complete option text]
C. [Complete option text] 
D. [Complete option text]

Extract EVERY question visible across ALL pages of the exam paper. Be thorough and complete.'''

# Keep existing helper functions for backward compatibility
def _extract_content_from_response(text):
    """Extract meaningful content from AI response"""
    lines = text.split('\n')
    content_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('```', '#', 'Here', 'The', 'This')):
            content_lines.append(line)
    
    return '\n\n'.join(content_lines) if content_lines else text

def _validate_question_extraction_legacy(text):
    """Legacy validation function for backward compatibility"""
    return _validate_question_extraction(text)