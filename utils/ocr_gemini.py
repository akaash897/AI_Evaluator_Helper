# utils/ocr_gemini.py - Enhanced version
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
        prompt = '''EXTRACT ALL EXAMINATION QUESTIONS

You are analyzing an academic examination paper. Extract EVERY question that students must answer.

REQUIREMENTS:

1. FIND ALL QUESTIONS:
   - Question numbers: "1.", "2.", "Q1", "Q2", "Question 1", etc.
   - Sub-parts: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
   - Multiple choice options: A., B., C., D.
   - Mark allocations: [2 marks], [10], (5), etc.

2. EXTRACT COMPLETE CONTENT:
   - Full question text with all details
   - Mathematical expressions and matrices
   - All sub-questions and parts
   - Multiple choice options with complete text
   - References to figures/diagrams

3. OUTPUT FORMAT:
Question 1: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next complete question] [marks if shown]
A. [Option A complete text]
B. [Option B complete text]  
C. [Option C complete text]
D. [Option D complete text]

4. IGNORE:
   - Institution headers and course codes
   - Exam duration, total marks, date/time
   - General instructions not part of questions
   - Page numbers and administrative text

CRITICAL: Extract the COMPLETE text for each question. Include all mathematical expressions, detailed descriptions, and instructions. Do not summarize or abbreviate.

Extract ALL questions in the specified format:'''

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    try:
        response = model.generate_content([prompt] + images)
        result = response.text.strip()
        
        # Validate and enhance the result
        result = _enhance_question_extraction(result)
        
        # Final validation
        if _validate_question_extraction(result):
            return result
        else:
            # Try again with more specific prompt
            print("Initial extraction insufficient, retrying with enhanced prompt...")
            enhanced_prompt = _create_enhanced_question_prompt()
            
            response = model.generate_content([enhanced_prompt] + images)
            result = response.text.strip()
            return _enhance_question_extraction(result)
            
    except Exception as e:
        print(f"Error in Gemini question extraction: {e}")
        return f"Error extracting questions: {str(e)}"

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
    return '''DETAILED QUESTION EXTRACTION - STEP BY STEP

Step 1: Scan the exam paper for question numbers
- Look for: "1.", "2.", "3.", "Q1", "Q2", "Question 1", etc.
- Note the location of each question number

Step 2: For each question found, extract:
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

Extract EVERY question visible in the exam paper. Be thorough and complete.'''

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