# utils/ocr_openai.py - Enhanced version
import os
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import base64
import openai
import re

def pdf_to_images(pdf_path):
    # Enhanced with higher DPI for better text recognition
    images = convert_from_path(pdf_path, dpi=350, fmt='png')
    image_paths = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(f"tmp/{base_name}", exist_ok=True)
    for i, img in enumerate(images):
        img_path = f"tmp/{base_name}/page_{i + 1}.png"
        img.save(img_path, "PNG", optimize=True, quality=95)
        image_paths.append(img_path)
    return image_paths

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def gpt4o_extract_answer_latex(image_paths, question_text, prompt=None):
    if prompt is None:
        prompt = f"""Generate a comprehensive LaTeX document that maps student answers to exam questions.

CRITICAL REQUIREMENTS:

1. COMPLETE LATEX DOCUMENT: Must start with \\documentclass and end with \\end{{document}}

2. STRUCTURE REQUIRED:
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
[Extract complete student response here]
\\end{{quote}}

\\vspace{{0.5cm}}

\\subsection*{{Question 2}}
\\textbf{{Question:}} [Next complete question]

\\textbf{{Student Answer:}}
\\begin{{quote}}
[Extract complete student response here]
\\end{{quote}}

[Continue for all questions...]

\\end{{document}}

3. EXTRACTION INSTRUCTIONS:
- Extract ALL visible handwritten content from the student answer sheet
- Match answers to question numbers when identifiable (look for Q1, 1., Question 1, etc.)
- Include mathematical calculations, diagrams, and all working
- For diagrams, describe them: "Student drew a graph showing..."
- Use proper LaTeX math environments for equations
- If no clear question numbers, extract content sequentially
- Preserve the student's exact work and reasoning
- Include crossed-out work and corrections
- Do NOT summarize or correct the student's work

4. QUESTION PAPER CONTENT:
{question_text}

5. MAPPING STRATEGY:
- Look for question numbers in both the question paper and answer sheet
- Match Q1 with Question 1, (1) with Question 1, etc.
- Group related work under the same question
- If uncertain about mapping, note: "Student appears to be answering: [topic]"

Generate ONLY the complete LaTeX document. Start with \\documentclass and end with \\end{{document}}."""
    
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
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"  # Enhanced for better text recognition
            }
        })
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=10000  # Increased for longer documents
        )
        
        latex_output = response.choices[0].message.content
        
        # Enhanced cleaning and validation
        latex_output = _enhanced_clean_openai_output(latex_output)
        
        # Validate structure
        if not _validate_openai_latex_structure(latex_output):
            print("Generated LaTeX failed validation, creating enhanced fallback...")
            latex_output = _create_openai_enhanced_fallback(latex_output, question_text)
        
        return latex_output
        
    except Exception as e:
        print(f"Error in OpenAI processing: {e}")
        return _create_openai_enhanced_fallback(f"Error: {str(e)}", question_text)

def gpt4o_extract_questions(image_paths, prompt=None):
    """Enhanced function for question extraction with GPT-4V"""
    
    if prompt is None:
        prompt = """COMPREHENSIVE QUESTION EXTRACTION FROM EXAMINATION PAPER

Extract ALL examination questions that students need to answer.

DETAILED REQUIREMENTS:

1. IDENTIFY ALL QUESTIONS:
   - Question numbers: "1.", "2.", "Q1", "Q2", "Question 1", "Question 2", etc.
   - Sub-questions: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
   - Multiple choice questions with options A, B, C, D
   - Mark allocations: [2 marks], [10], (5), etc.

2. EXTRACT COMPLETE CONTENT:
   - Full question text with all details and instructions
   - Mathematical expressions, matrices, and formulas exactly as shown
   - All sub-parts and their complete text
   - Complete text for all multiple choice options
   - References to figures, diagrams, or tables

3. PRESERVE STRUCTURE:
   - Maintain question hierarchy and numbering
   - Keep proper indentation for sub-parts
   - Include all instructional text that's part of the question
   - Preserve mathematical notation exactly

4. OUTPUT FORMAT:
Question 1: [Complete question text including all details] [marks if shown]
(a) [Complete sub-question text]
(b) [Complete sub-question text]

Question 2: [Next complete question with all details] [marks if shown]  
A. [Complete option A text]
B. [Complete option B text]
C. [Complete option C text]
D. [Complete option D text]

Question 3: [Continue for all questions...]

5. IGNORE ADMINISTRATIVE CONTENT:
   - Institution headers, course codes, instructor names
   - Exam instructions, duration, total marks
   - Page numbers, footers, watermarks
   - General instructions not part of specific questions

6. CRITICAL: Extract the COMPLETE question text. Include mathematical expressions, detailed descriptions, and all instructions. Do not summarize or abbreviate questions.

Extract ALL questions in the specified format. Be thorough and comprehensive."""

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
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"
            }
        })
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=3000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Enhanced validation and processing
        result = _enhance_openai_question_extraction(result)
        
        # Validate the extraction
        if _is_valid_openai_question_extraction(result):
            return result
        else:
            # Retry with more specific prompt
            print("Initial extraction insufficient, retrying...")
            enhanced_prompt = _create_openai_enhanced_question_prompt()
            
            messages[0]["content"][0]["text"] = enhanced_prompt
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                max_tokens=10000
            )
            
            result = response.choices[0].message.content.strip()
            return _enhance_openai_question_extraction(result)
            
    except Exception as e:
        print(f"Error in OpenAI question extraction: {e}")
        return f"Error extracting questions: {str(e)}"

def _enhanced_clean_openai_output(latex_output):
    """Enhanced cleaning for OpenAI LaTeX output"""
    if not latex_output:
        return ""
    
    # Remove markdown code blocks
    if "```latex" in latex_output:
        latex_output = latex_output.split("```latex")[1].split("```")[0].strip()
    elif "```" in latex_output:
        parts = latex_output.split("```")
        if len(parts) >= 3:
            latex_output = parts[1].strip()
    
    # Clean up common formatting issues
    latex_output = latex_output.replace("\\textbf{Question:}", "\\textbf{Question:}")
    latex_output = latex_output.replace("\\textbf{Student Answer:}", "\\textbf{Student Answer:}")
    
    # Ensure proper document structure
    if not latex_output.startswith("\\documentclass"):
        # Try to extract valid LaTeX from the response
        doc_match = re.search(r'\\documentclass.*?\\end\{document\}', latex_output, re.DOTALL)
        if doc_match:
            latex_output = doc_match.group(0)
    
    return latex_output.strip()

def _validate_openai_latex_structure(latex_output):
    """Validate LaTeX structure for OpenAI output"""
    required_elements = [
        "\\documentclass",
        "\\begin{document}",
        "\\end{document}",
        "\\title{",
        "\\maketitle"
    ]
    
    return all(element in latex_output for element in required_elements)

def _create_openai_enhanced_fallback(content, question_text):
    """Create enhanced fallback document for OpenAI"""
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

\\section*{{Extracted Student Content}}
\\begin{{quote}}
{content[:1500] if content else "No clear content extracted"}
\\end{{quote}}

\\section*{{Technical Information}}
\\begin{{itemize}}
\\item AI Model: OpenAI GPT-4 Vision
\\item Processing Status: Partial extraction
\\item Issue: LaTeX structure validation failed
\\item Recommendation: Review source documents and retry
\\end{{itemize}}

\\end{{document}}"""

def _enhance_openai_question_extraction(text):
    """Enhance OpenAI question extraction result"""
    if not text or len(text.strip()) < 50:
        return text
    
    lines = text.split('\n')
    enhanced_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Enhance question number formatting
            if re.match(r'^\d+[\.:]\s', line) and not line.startswith('Question'):
                line = f"Question {line}"
            enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)

def _is_valid_openai_question_extraction(text):
    """Validate OpenAI question extraction"""
    if not text or len(text.strip()) < 100:
        return False
        
    # Check for question patterns
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
    
    # Check for reasonable length and content
    has_sufficient_content = len(text.strip()) > 100
    
    # Check it's not mostly administrative content
    admin_indicators = ['total marks', 'duration', 'time allowed', 'instructions']
    admin_heavy = sum(1 for indicator in admin_indicators if indicator in text.lower()) > 2
    
    return has_questions and has_sufficient_content and not admin_heavy

def _create_openai_enhanced_question_prompt():
    """Create enhanced question extraction prompt for OpenAI"""
    return """STEP-BY-STEP QUESTION EXTRACTION

Follow these steps to extract ALL questions from the exam paper:

Step 1: SCAN FOR QUESTION NUMBERS
- Look for: "1.", "2.", "3.", "Q1", "Q2", "Question 1", etc.
- Note each question's location and number

Step 2: EXTRACT COMPLETE QUESTIONS
For each question found:
- Extract the complete question text
- Include all sub-parts: (a), (b), (c) or (i), (ii), (iii)
- Include mark allocations: [2], [10 marks], etc.
- For multiple choice, include ALL options A, B, C, D

Step 3: PRESERVE MATHEMATICAL CONTENT
- Copy mathematical expressions exactly
- Include matrices, equations, and formulas
- Note references to figures or diagrams

Step 4: FORMAT OUTPUT
Question 1: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next question] [marks if shown]
A. [Option A text]
B. [Option B text]
C. [Option C text]
D. [Option D text]

Step 5: VERIFY COMPLETENESS
- Ensure all visible questions are extracted
- Check that mathematical expressions are complete
- Verify sub-parts are included

Extract EVERY question completely and accurately."""

# Legacy helper functions for backward compatibility
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