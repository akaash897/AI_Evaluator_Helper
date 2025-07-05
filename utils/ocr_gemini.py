# utils/ocr_gemini.py
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
        prompt = f"""Create a professional LaTeX document matching student answers with questions.

REQUIRED OUTPUT FORMAT:
\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\setlength{{\\parskip}}{{6pt}}

\\begin{{document}}

\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing}}
\\date{{\\today}}
\\maketitle

\\section*{{Student Responses}}

[For each question, create sections like this:]

\\section*{{Question 1}}
\\textbf{{Student Answer:}}
\\begin{{quote}}
[Complete student response exactly as written]
\\end{{quote}}

\\vspace{{1cm}}

[Continue for all questions...]

\\end{{document}}

EXTRACTION RULES:
1. Extract ALL visible student handwriting
2. Match answers to question numbers when possible
3. Include mathematical work and calculations
4. Describe diagrams as [Student drew: description]
5. If no clear question numbers, extract all content sequentially
6. Use proper LaTeX math formatting

QUESTION CONTEXT:
{question_text}

Generate ONLY the complete LaTeX document starting with \\documentclass."""

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    response = model.generate_content([prompt] + images)
    
    # Clean and validate LaTeX output
    latex_text = response.text.strip()
    
    # Remove any markdown code blocks
    if "```latex" in latex_text:
        latex_text = latex_text.split("```latex")[1].split("```")[0].strip()
    elif "```" in latex_text:
        parts = latex_text.split("```")
        if len(parts) >= 3:
            latex_text = parts[1].strip()
    
    # Ensure proper LaTeX structure
    if not latex_text.startswith("\\documentclass"):
        # If missing document structure, wrap content properly
        wrapped_content = _extract_content_from_response(latex_text)
        latex_text = f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\setlength{{\\parskip}}{{6pt}}

\\begin{{document}}

\\title{{Student Answer Sheet}}
\\author{{Automated Processing}}
\\date{{\\today}}
\\maketitle

\\section*{{Extracted Student Work}}

{wrapped_content}

\\end{{document}}"""
    
    return latex_text

def gemini_extract_question_text(image_paths, prompt=None):
    configure_gemini()
    
    if prompt is None:
        prompt = '''EXTRACT EXAMINATION QUESTIONS ONLY

You are analyzing an academic examination paper. Your task is to find and extract ONLY the actual questions that students need to answer.

WHAT TO LOOK FOR:
1. Question numbers: "1.", "Q1", "Question 1", "(1)", "1)", etc.
2. Question content: Text that asks something or requires a solution
3. Command words: "What", "How", "Explain", "Calculate", "Solve", "Find", "Describe"
4. Mathematical problems with equations or variables
5. Multiple choice options (A, B, C, D)
6. Mark allocations: [5 marks], (10), [3], etc.

WHAT TO IGNORE:
- Institution names, headers, logos
- Course codes, instructor names
- Exam duration, total marks, date/time
- General instructions ("Answer all questions", "Write clearly")
- Page numbers, footers
- Administrative text

OUTPUT FORMAT:
Question 1: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next question...]

VALIDATION:
- Each line should contain actual question content
- Questions should be numbered
- Include complete question text, not fragments
- Preserve mathematical notation exactly

If no clear questions are found, respond: "NO VALID QUESTIONS DETECTED"

Focus on identifying question patterns and extracting complete question text.'''

    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path))
    
    response = model.generate_content([prompt] + images)
    result = response.text.strip()
    
    # Post-process to ensure we got actual questions
    if _validate_question_extraction(result):
        return result
    else:
        # Try again with more specific prompt
        enhanced_prompt = """Look at this exam paper image very carefully. 

FIND THE QUESTIONS:
1. Scan for numbers followed by text: "1.", "2.", "Q1", "Question 1"
2. Look for question words: What, How, Why, Explain, Calculate, Solve
3. Ignore headers, instructions, and administrative text

EXTRACT FORMAT:
Question 1: [full question text]
Question 2: [full question text]

If you cannot find numbered questions, look for any text that asks students to do something or solve something. Extract that content.

Be very careful to distinguish between actual questions and other content like instructions or headers."""
        
        response = model.generate_content([enhanced_prompt] + images)
        return response.text.strip()

def _extract_content_from_response(text):
    """Extract meaningful content from AI response"""
    lines = text.split('\n')
    content_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('```', '#', 'Here', 'The', 'This')):
            content_lines.append(line)
    
    return '\n\n'.join(content_lines) if content_lines else text

def _validate_question_extraction(text):
    """Validate that extracted text contains actual questions"""
    if not text or len(text.strip()) < 20:
        return False
    
    # Check for question indicators
    question_patterns = [
        r'Question\s+\d+',
        r'Q\d+',
        r'^\d+[\.:]\s',
        r'\(\d+\)',
        r'\d+\)'
    ]
    
    has_question_numbers = any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) for pattern in question_patterns)
    
    # Check for question words
    question_words = ['what', 'how', 'why', 'explain', 'calculate', 'solve', 'find', 'describe']
    has_question_words = any(word in text.lower() for word in question_words)
    
    # Check it's not just numbers/matrices
    lines = text.strip().split('\n')
    numeric_lines = sum(1 for line in lines if re.match(r'^\s*[\d\s\.\-\+]+\s*$', line.strip()))
    mostly_numeric = numeric_lines > len(lines) * 0.7
    
    return (has_question_numbers or has_question_words) and not mostly_numeric