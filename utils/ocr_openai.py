# utils/ocr_openai.py
import os
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import base64
import openai
import re

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(f"tmp/{base_name}", exist_ok=True)
    for i, img in enumerate(images):
        img_path = f"tmp/{base_name}/page_{i + 1}.png"
        img.save(img_path, "PNG")
        image_paths.append(img_path)
    return image_paths

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def gpt4o_extract_answer_latex(image_paths, question_text, prompt=None):
    if prompt is None:
        prompt = f"""Generate a complete LaTeX document matching student answers with questions.

EXACT REQUIRED FORMAT:
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

\\section*{{Student Responses}}

\\section*{{Question 1}}
\\textbf{{Student Answer:}}
\\begin{{quote}}
[Extract complete student response here]
\\end{{quote}}

\\vspace{{1cm}}

\\section*{{Question 2}}
\\textbf{{Student Answer:}}
\\begin{{quote}}
[Extract complete student response here]
\\end{{quote}}

[Continue for all questions...]

\\end{{document}}

EXTRACTION INSTRUCTIONS:
1. Extract ALL visible handwritten content from the student answer sheet
2. Match answers to question numbers when identifiable
3. Include mathematical calculations, diagrams, and all working
4. For diagrams, describe them: "Student drew a graph showing..."
5. Use proper LaTeX math environments for equations
6. If no clear question numbers, extract content sequentially
7. Preserve the student's exact work and reasoning

QUESTION CONTEXT:
{question_text}

Generate ONLY the LaTeX document. Start with \\documentclass and end with \\end{{document}}."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    for path in image_paths:
        img_b64 = encode_image_base64(path)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }
        })
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1
    )
    
    latex_output = response.choices[0].message.content
    
    # Clean and validate output
    latex_output = latex_output.strip()
    
    # Remove markdown if present
    if "```latex" in latex_output:
        latex_output = latex_output.split("```latex")[1].split("```")[0].strip()
    elif "```" in latex_output:
        parts = latex_output.split("```")
        if len(parts) >= 3:
            latex_output = parts[1].strip()
    
    # Ensure proper document structure
    if not latex_output.startswith("\\documentclass"):
        content = _extract_meaningful_content(latex_output)
        latex_output = f"""\\documentclass[12pt]{{article}}
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

{content}

\\end{{document}}"""
    
    return latex_output

def gpt4o_extract_questions(image_paths, prompt=None):
    """Dedicated function for question extraction with GPT-4V"""
    
    if prompt is None:
        prompt = """Extract ONLY the examination questions from this academic paper.

TASK: Identify and transcribe actual questions that students need to answer.

LOOK FOR:
- Question numbers: "1.", "Q1", "Question 1:", "(1)", "1)"
- Question indicators: "What", "How", "Why", "Explain", "Calculate", "Solve", "Find"
- Mathematical problems with equations or variables
- Multiple choice questions with options A, B, C, D
- Mark allocations: [5 marks], (10), [3], etc.

IGNORE:
- Institution headers, course codes, instructor names
- Exam instructions, duration, total marks
- "Answer all questions", "Write clearly", etc.
- Page numbers, footers, administrative text

OUTPUT FORMAT:
Question 1: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next complete question...]

REQUIREMENTS:
- Each question must be numbered
- Include complete question text
- Preserve mathematical notation exactly
- If no questions found, state: "NO QUESTIONS DETECTED"

Extract only the actual questions, not instructions or headers."""

    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    for path in image_paths:
        img_b64 = encode_image_base64(path)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }
        })
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1
    )
    
    result = response.choices[0].message.content.strip()
    
    # Validate the extraction
    if _is_valid_question_extraction(result):
        return result
    else:
        # Retry with enhanced prompt
        enhanced_prompt = """Look at this exam paper very carefully. I need you to find the actual QUESTIONS that students need to answer.

Step 1: Scan the image for question numbers like "1.", "2.", "Q1", "Question 1"
Step 2: Look for question words: What, How, Why, Explain, Calculate, Solve, Find, Describe
Step 3: Ignore headers, instructions, and administrative text

Extract the questions in this format:
Question 1: [full question text]
Question 2: [full question text]

Focus ONLY on extracting the questions themselves, not instructions or other content."""
        
        messages[0]["content"][0]["text"] = enhanced_prompt
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()

def _extract_meaningful_content(text):
    """Extract meaningful content from response, filtering out explanations"""
    lines = text.split('\n')
    content_lines = []
    
    skip_phrases = ['Here is', 'The document', 'I have', 'This is', 'Based on']
    
    for line in lines:
        line = line.strip()
        if line and not any(line.startswith(phrase) for phrase in skip_phrases):
            content_lines.append(line)
    
    return '\n\n'.join(content_lines) if content_lines else text

def _is_valid_question_extraction(text):
    """Validate that the extracted text contains actual questions"""
    if not text or len(text.strip()) < 20:
        return False
        
    # Check for question patterns
    question_patterns = [
        r'Question\s+\d+',
        r'Q\d+',
        r'^\d+[\.:]\s',
        r'\(\d+\)',
        r'\d+\)'
    ]
    
    has_question_numbers = any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) 
                              for pattern in question_patterns)
    
    # Check for question words
    question_words = ['what', 'how', 'why', 'explain', 'calculate', 'solve', 'find', 'describe', 'determine']
    has_question_words = any(word in text.lower() for word in question_words)
    
    # Check it's not mostly numeric data
    lines = text.strip().split('\n')
    numeric_lines = sum(1 for line in lines if re.match(r'^\s*[\d\s\.\-\+]+\s*$', line.strip()))
    mostly_numeric = numeric_lines > len(lines) * 0.6
    
    # Check for common non-question content
    non_question_indicators = ['total marks', 'duration', 'time allowed', 'instructions']
    has_admin_content = any(indicator in text.lower() for indicator in non_question_indicators)
    
    return (has_question_numbers or has_question_words) and not mostly_numeric and not has_admin_content