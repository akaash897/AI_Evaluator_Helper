# utils/ocr_openai.py - Enhanced version with multi-page support
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
    
    print(f"DEBUG: Converted PDF to {len(image_paths)} images")
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
    """Enhanced function for multi-page question extraction with GPT-4V"""
    
    if prompt is None:
        prompt = """COMPREHENSIVE MULTI-PAGE QUESTION EXTRACTION

CRITICAL: This examination paper has MULTIPLE PAGES. You must extract questions from ALL pages.

Extract ALL examination questions from the ENTIRE document (all pages provided).

MULTI-PAGE PROCESSING INSTRUCTIONS:
1. EXAMINE ALL IMAGES: Look through every page/image provided
2. EXTRACT FROM EVERY PAGE: Find questions on all pages, not just the first
3. MAINTAIN CONTINUITY: Questions may continue across pages
4. PRESERVE ORDER: Extract questions in sequence across all pages

DETAILED REQUIREMENTS:

1. IDENTIFY ALL QUESTIONS ACROSS ALL PAGES:
   - Question numbers: "1.", "2.", "Q1", "Q2", "Question 1", "Question 2", etc.
   - Sub-questions: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
   - Multiple choice questions with options A, B, C, D
   - Mark allocations: [2 marks], [10], (5), etc.

2. EXTRACT COMPLETE CONTENT FROM ALL PAGES:
   - Full question text with all details and instructions
   - Mathematical expressions, matrices, and formulas exactly as shown
   - All sub-parts and their complete text
   - Complete text for all multiple choice options
   - References to figures, diagrams, or tables

3. PRESERVE STRUCTURE ACROSS PAGES:
   - Maintain question hierarchy and numbering
   - Keep proper indentation for sub-parts
   - Include all instructional text that's part of the question
   - Preserve mathematical notation exactly

4. OUTPUT FORMAT FOR ALL PAGES:
Question 1: [Complete question text including all details] [marks if shown]
(a) [Complete sub-question text]
(b) [Complete sub-question text]

Question 2: [Next complete question with all details] [marks if shown]  
A. [Complete option A text]
B. [Complete option B text]
C. [Complete option C text]
D. [Complete option D text]

Question 3: [Continue for ALL questions from ALL pages...]

5. IGNORE ADMINISTRATIVE CONTENT ON ALL PAGES:
   - Institution headers, course codes, instructor names
   - Exam instructions, duration, total marks
   - Page numbers, footers, watermarks
   - General instructions not part of specific questions

6. CRITICAL: Extract the COMPLETE question text from ALL pages. Include mathematical expressions, detailed descriptions, and all instructions. Do not summarize or abbreviate questions.

Process ALL pages and extract ALL questions in the specified format. Be thorough and comprehensive across the entire document."""

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
    
    # Add ALL images to the request
    for i, path in enumerate(image_paths):
        img_b64 = encode_image_base64(path)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"
            }
        })
    
    print(f"DEBUG: Sending {len(image_paths)} pages to OpenAI for question extraction")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=10000  # Increased for multi-page content
        )
        
        result = response.choices[0].message.content.strip()
        print(f"DEBUG: OpenAI returned {len(result)} characters for {len(image_paths)} pages")
        
        # Enhanced validation and processing for multi-page
        result = _enhance_openai_multi_page_extraction(result, len(image_paths))
        
        # Validate the extraction
        if _is_valid_openai_multi_page_extraction(result, len(image_paths)):
            return result
        else:
            # Retry with page-by-page approach
            print("Multi-page extraction validation failed, trying page-by-page...")
            return _openai_extract_page_by_page(image_paths, prompt)
            
    except Exception as e:
        print(f"Error in OpenAI question extraction: {e}")
        return f"Error extracting questions: {str(e)}"

def _openai_extract_page_by_page(image_paths, base_prompt):
    """Fallback: Extract from each page individually"""
    all_questions = []
    
    for i, path in enumerate(image_paths):
        page_prompt = f"""EXTRACT QUESTIONS FROM PAGE {i+1}

This is page {i+1} of a {len(image_paths)}-page exam.

Extract ALL questions from this specific page. Continue question numbering appropriately.

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
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": page_prompt},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_base64(path)}",
                        "detail": "high"
                    }
                }
            ]
        }]
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                max_tokens=3000
            )
            
            page_result = response.choices[0].message.content.strip()
            if page_result and len(page_result) > 50:
                all_questions.append(f"\n=== PAGE {i+1} ===")
                all_questions.append(page_result)
                print(f"DEBUG: Page {i+1} extracted {len(page_result)} characters")
            else:
                print(f"DEBUG: Page {i+1} had minimal content")
                
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
    
    combined_result = "\n".join(all_questions)
    print(f"DEBUG: Combined result from all pages: {len(combined_result)} characters")
    return combined_result

def _is_valid_openai_multi_page_extraction(text, num_pages):
    """Validate OpenAI multi-page extraction"""
    if not text or len(text.strip()) < 100:
        print("DEBUG: OpenAI validation failed - text too short")
        return False
        
    # For multi-page documents, expect more content
    if num_pages > 1:
        expected_min_length = num_pages * 150
        if len(text) < expected_min_length:
            print(f"DEBUG: OpenAI extracted content too short for {num_pages} pages")
            return False
    
    # Check for question distribution
    import re
    question_count = len(re.findall(r'Question\s+\d+', text, re.IGNORECASE))
    
    if num_pages > 2 and question_count < num_pages:
        print(f"DEBUG: Only {question_count} questions found across {num_pages} pages")
        # Don't fail here - some pages might have fewer questions
    
    print(f"DEBUG: OpenAI multi-page validation passed - {question_count} questions in {num_pages} pages")
    return True

def _enhance_openai_multi_page_extraction(text, num_pages):
    """Enhance OpenAI extraction for multi-page validation"""
    return text  # OpenAI usually handles this well

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
    return """STEP-BY-STEP MULTI-PAGE QUESTION EXTRACTION

Follow these steps to extract ALL questions from the COMPLETE exam paper:

Step 1: SCAN FOR QUESTION NUMBERS ON ALL PAGES
- Look for: "1.", "2.", "3.", "Q1", "Q2", "Question 1", etc.
- Note each question's location and number on each page
- Process every page provided, not just the first

Step 2: EXTRACT COMPLETE QUESTIONS FROM ALL PAGES
For each question found on any page:
- Extract the complete question text
- Include all sub-parts: (a), (b), (c) or (i), (ii), (iii)
- Include mark allocations: [2], [10 marks], etc.
- For multiple choice, include ALL options A, B, C, D

Step 3: PRESERVE MATHEMATICAL CONTENT FROM ALL PAGES
- Copy mathematical expressions exactly
- Include matrices, equations, and formulas
- Note references to figures or diagrams

Step 4: FORMAT OUTPUT FOR COMPLETE DOCUMENT
Question 1: [Complete question text] [marks if shown]
(a) [Sub-question if any]
(b) [Sub-question if any]

Question 2: [Next question] [marks if shown]
A. [Option A text]
B. [Option B text]
C. [Option C text]
D. [Option D text]

Step 5: VERIFY COMPLETENESS ACROSS ALL PAGES
- Ensure all visible questions from all pages are extracted
- Check that mathematical expressions are complete
- Verify sub-parts are included

Extract EVERY question from EVERY page completely and accurately."""

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