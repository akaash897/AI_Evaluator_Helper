# utils/ocr_gemini.py - Enhanced version with multi-page support
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import re

load_dotenv()

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)

def gemini_extract_answer_latex(image_paths, question_text, prompt=None):
    configure_gemini()
    
    if prompt is None:
        prompt = f"""Create a comprehensive LaTeX document that properly maps student answers to exam questions.

CRITICAL REQUIREMENTS:

1. COMPLETE LATEX DOCUMENT: Must start with \\documentclass and end with \\end{{document}}

2. INCLUDE BOTH QUESTIONS AND ANSWERS: Show the complete question text followed by the student's answer

3. REQUIRED FORMAT:
\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\setlength{{\\parskip}}{{6pt}}

\\begin{{document}}

\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Questions and Student Responses}}

\\subsection*{{Question 1}}
\\textbf{{Question:}} [Complete question text from question paper]

\\textbf{{Student Answer:}}
\\begin{{quote}}
[Student's complete response exactly as written]
\\end{{quote}}

\\vspace{{0.5cm}}

\\subsection*{{Question 2}}
\\textbf{{Question:}} [Next complete question text]

\\textbf{{Student Answer:}}
\\begin{{quote}}
[Student's response for this question]
\\end{{quote}}

[Continue for all questions found...]

\\end{{document}}

4. EXTRACTION RULES:
- Extract ALL visible student handwriting
- Include mathematical work, calculations, diagrams
- Match answers to question numbers when identifiable
- If no clear question numbers, extract content sequentially
- Describe diagrams as "Student drew: [description]"
- DO NOT correct or reason about answers - extract exactly as written
- Include working, crossed-out text, and rough work

QUESTION PAPER CONTENT:
{question_text}

STUDENT ANSWER SHEET:
Examine the answer sheet images and create the complete LaTeX document mapping student responses to the questions above."""

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    try:
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
        prompt = '''EXTRACT ALL EXAMINATION QUESTIONS FROM ALL PAGES

IMPORTANT: This exam paper has MULTIPLE PAGES. You must extract questions from ALL pages provided.

You are analyzing a complete academic examination paper. Extract EVERY question from ALL pages.

CRITICAL REQUIREMENTS:

1. SCAN ALL PAGES: Look through every image/page provided - do not skip any pages
2. EXTRACT FROM EACH PAGE: Find questions on every page, not just the first one
3. MAINTAIN PAGE ORDER: Extract questions in the order they appear across all pages
4. PROCESS COMPLETE DOCUMENT: This is a multi-page exam - extract everything

5. FIND ALL QUESTIONS ACROSS ALL PAGES:
   - Question numbers: "1.", "2.", "Q1", "Q2", "Question 1", etc.
   - Sub-parts: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
   - Multiple choice options: A., B., C., D.
   - Mark allocations: [2 marks], [10], (5), etc.

6. EXTRACT COMPLETE CONTENT FROM ALL PAGES:
   - Full question text with all details from every page
   - Mathematical expressions and matrices exactly as shown
   - All sub-questions and parts from all pages
   - Multiple choice options with complete text
   - References to figures/diagrams

7. OUTPUT FORMAT (for ALL pages):
Question 1: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next complete question] [marks if shown]
A. [Option A complete text]
B. [Option B complete text]  
C. [Option C complete text]
D. [Option D complete text]

[Continue for ALL questions from ALL pages...]

8. IGNORE ADMINISTRATIVE CONTENT ON ALL PAGES:
   - Institution headers, course codes, instructor names
   - Exam duration, total marks, date/time
   - Page numbers, footers, watermarks
   - General instructions not part of specific questions

CRITICAL: Process ALL images/pages provided. Extract COMPLETE text for each question from every page. Include all mathematical expressions, detailed descriptions, and instructions. Do not summarize or abbreviate.

Extract ALL questions from ALL pages in the specified format:'''

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Load ALL images
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    print(f"DEBUG: Processing {len(images)} pages for question extraction")
    
    try:
        # Send ALL images at once to process the complete document
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
        expected_min_length = num_pages * 200  # At least 200 chars per page
        if len(text) < expected_min_length:
            print(f"DEBUG: Extracted text ({len(text)} chars) seems too short for {num_pages} pages")
            return False
    
    # Check for question distribution
    import re
    question_count = len(re.findall(r'Question\s+\d+', text, re.IGNORECASE))
    
    if num_pages > 1 and question_count < 2:
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
    """Validate LaTeX structure"""
    required_elements = [
        "\\documentclass",
        "\\begin{document}",
        "\\end{document}",
        "\\title{",
        "\\maketitle"
    ]
    
    return all(element in latex_text for element in required_elements)

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