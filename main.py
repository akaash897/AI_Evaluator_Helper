import os
import traceback
import asyncio
import subprocess
import re
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Keep your original utility imports
from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex, gpt4o_extract_questions
from utils.ocr_gemini import gemini_extract_answer_latex, gemini_extract_question_text

# Import agentic components
try:
    from agents import ExamProcessingOrchestrator, DocumentAnalyzerAgent, QuestionExtractorAgent
    AGENTIC_AVAILABLE = True
    print("Agentic system loaded successfully")
except ImportError as e:
    print(f"Agentic system not available: {e}")
    print("Falling back to original methods")
    AGENTIC_AVAILABLE = False

STUDENT_PDF_FOLDER = "uploads/students_data"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Ensure API keys are set
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    print("OpenAI API key loaded")
else:
    print("Warning: OPENAI_API_KEY not found in environment")

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print("Gemini API key loaded")
else:
    print("Warning: GEMINI_API_KEY not found in environment")

# Initialize the agentic orchestrator if available
if AGENTIC_AVAILABLE:
    try:
        orchestrator = ExamProcessingOrchestrator()
        print("Agentic orchestrator initialized")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        AGENTIC_AVAILABLE = False

def extract_question_text(pdf_path: str, fallback_model: str = "gemini"):
    """Extract questions using agentic system with proper model selection"""
    try:
        if AGENTIC_AVAILABLE:
            print("🤖 Using agentic system for question extraction...")
            
            # Run the agentic analysis and extraction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create analyzer and extractor agents
                analyzer = DocumentAnalyzerAgent()
                extractor = QuestionExtractorAgent()
                
                # Analyze document first
                analysis_task = {
                    "file_path": pdf_path,
                    "file_type": "question_paper"
                }
                analysis_result = loop.run_until_complete(analyzer.execute(analysis_task))
                
                if not analysis_result.success:
                    print(f"❌ Analysis failed: {analysis_result.error}")
                    raise Exception(f"Document analysis failed: {analysis_result.error}")
                
                # Use the agentic system's recommendation - NO OVERRIDE
                strategy = analysis_result.data["strategy"]
                recommended_model = strategy["recommended_model"]
                print(f"🎯 Agentic system recommends: {recommended_model}")
                
                # Extract questions using the recommended model
                extraction_task = {
                    "file_path": pdf_path,
                    "strategy": strategy
                }
                extraction_result = loop.run_until_complete(extractor.execute(extraction_task))
                
                if extraction_result.success:
                    print("✅ Agentic extraction successful!")
                    question_text = extraction_result.data["question_text"]
                    pages_processed = extraction_result.data.get("pages_processed", 1)
                    
                    print(f"📊 Extraction Summary:")
                    print(f"   • Pages processed: {pages_processed}")
                    print(f"   • Total characters: {len(question_text)}")
                    print(f"   • Model used: {extraction_result.data['model_used']}")
                    print(f"   • Confidence: {extraction_result.confidence:.2f}")
                    
                    # Enhanced validation for multi-page content
                    validation_result = _validate_extracted_questions(question_text, pages_processed)
                    
                    if not validation_result["is_valid"]:
                        print("⚠️ Validation failed, retrying with fallback model...")
                        strategy["recommended_model"] = fallback_model
                        extraction_task["strategy"] = strategy
                        retry_result = loop.run_until_complete(extractor.execute(extraction_task))
                        if retry_result.success:
                            question_text = retry_result.data["question_text"]
                            print(f"✅ Fallback extraction successful with {fallback_model}")
                        else:
                            raise Exception("Both agentic and fallback extraction failed")
                    
                    return question_text
                else:
                    print(f"❌ Agentic extraction failed: {extraction_result.error}")
                    raise Exception(f"Agentic extraction failed: {extraction_result.error}")
                    
            finally:
                loop.close()
                
        else:
            raise Exception("Agentic system not available")
            
    except Exception as e:
        print(f"❌ Agentic extraction failed, using enhanced fallback method: {e}")
        return _enhanced_extract_question_text(pdf_path, fallback_model)

def _validate_extracted_questions(question_text: str, num_pages: int) -> dict:
    """Validate extracted questions for multi-page completeness"""
    validation = {
        "is_valid": True,
        "issues": []
    }
    
    if not question_text or len(question_text.strip()) < 100:
        validation["is_valid"] = False
        validation["issues"].append("Text too short")
        return validation
    
    # Check for reasonable content per page
    if num_pages > 1:
        expected_min_length = num_pages * 150
        if len(question_text) < expected_min_length:
            validation["is_valid"] = False
            validation["issues"].append(f"Insufficient content for {num_pages} pages")
    
    # Check for question patterns
    import re
    question_count = len(re.findall(r'Question\s+\d+', question_text, re.IGNORECASE))
    
    if question_count == 0:
        validation["is_valid"] = False
        validation["issues"].append("No questions detected")
    elif num_pages > 1 and question_count < 2:
        validation["is_valid"] = False
        validation["issues"].append("Too few questions for multi-page document")
    
    return validation

def _enhanced_extract_question_text(pdf_path: str, model: str = "gemini"):
    """Enhanced question extraction method with multi-page support"""
    try:
        print(f"📄 Converting PDF to images: {pdf_path}")
        image_paths = pdf_to_images(pdf_path)
        print(f"🖼️ Generated {len(image_paths)} images from PDF")
        
        enhanced_prompt = '''COMPREHENSIVE MULTI-PAGE QUESTION EXTRACTION

You are extracting questions from a complete academic examination that may span multiple pages.

CRITICAL: Process ALL pages provided. This examination paper has multiple pages - extract questions from EVERY page.

WHAT TO FIND ACROSS ALL PAGES:
1. Question numbers: "1.", "2.", "Q1", "Q2", "Question 1", "Question 2", etc.
2. Sub-questions: (a), (b), (c), (i), (ii), (iii), (1), (2), (3)
3. Mark allocations: [2 marks], [10], (5), etc.
4. Multiple choice options: A., B., C., D. with full text
5. All question content including:
   - Complete question statements
   - Mathematical expressions and matrices
   - References to figures/diagrams
   - Instructions within questions

FORMAT REQUIREMENTS FOR ALL PAGES:
Question 1: [Complete question text including all details] [marks if shown]
(a) [Sub-question text if any]
(b) [Sub-question text if any]

Question 2: [Next complete question with all details] [marks if shown]
A. [Option A text]
B. [Option B text]
C. [Option C text]  
D. [Option D text]

Question 3: [Continue for all questions from all pages...]

IGNORE ON ALL PAGES:
- Header information (Institution name, course codes)
- Exam metadata (Date, time, duration, total marks)
- General instructions not part of specific questions
- Page numbers and footers

CRITICAL: Extract the COMPLETE question text for each question from ALL pages. Include mathematical expressions, all sub-parts, and detailed descriptions. Do not summarize or shorten questions.

Process ALL pages and output only the extracted questions in the specified format.'''
        
        print(f"🤖 Sending {len(image_paths)} pages to {model.upper()} for enhanced question extraction...")
        
        if model == "gemini":
            result = gemini_extract_question_text(image_paths, enhanced_prompt)
        else:
            result = gpt4o_extract_questions(image_paths, enhanced_prompt)
        
        print(f"📝 {model.upper()} returned {len(result)} characters")
        
        # Post-process and validate result
        if not result or len(result.strip()) < 50:
            print("⚠️ First attempt produced insufficient content, trying alternative approach...")
            
            # Try with more specific prompt
            fallback_prompt = '''Extract exam questions step by step from ALL pages:

1. Look for numbered items across all pages: 1., 2., 3., Q1, Q2, Question 1, etc.
2. Extract the complete text for each numbered question from every page
3. Include any sub-parts like (a), (b), (c)
4. Include mark allocations like [2], [10 marks], (5 marks)
5. For multiple choice, include all A, B, C, D options

Format as:
Question 1: [Full question from page where found]
Question 2: [Full question from page where found] 
etc.

Extract ALL questions from ALL pages completely and accurately.'''
            
            if model == "gemini":
                result = gemini_extract_question_text(image_paths, fallback_prompt)
            else:
                result = gpt4o_extract_questions(image_paths, fallback_prompt)
            
            print(f"📝 Fallback attempt returned {len(result)} characters")
        
        # Final validation
        final_validation = _validate_extracted_questions(result, len(image_paths))
        if final_validation["is_valid"]:
            print("✅ Enhanced question extraction complete and validated")
        else:
            print(f"⚠️ Validation issues: {final_validation['issues']}")
        
        return result if result and len(result.strip()) > 50 else "Error: Could not extract questions from the provided PDF"
        
    except Exception as e:
        print(f"❌ Error in _enhanced_extract_question_text: {e}")
        print(traceback.format_exc())
        return f"Error extracting questions: {str(e)}"

def process_student_pdf(filename: str, question_text: str, output_folder: str, fallback_model: str = "gemini"):
    """Process student PDF with agentic system - using proper model selection"""
    try:
        student_name = os.path.splitext(filename)[0]
        local_path = os.path.join("uploads/students_data", filename)
        
        if not os.path.exists(local_path):
            print(f"❌ File not found: {local_path}")
            return None
        
        if AGENTIC_AVAILABLE:
            print(f"🤖 Processing {student_name} with agentic system...")
            
            # Use agentic processing with proper model selection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create analyzer and answer processor agents
                analyzer = DocumentAnalyzerAgent()
                
                # Analyze student document
                analysis_task = {
                    "file_path": local_path,
                    "file_type": "answer_sheet"
                }
                analysis_result = loop.run_until_complete(analyzer.execute(analysis_task))
                
                if analysis_result.success:
                    strategy = analysis_result.data["strategy"]
                    recommended_model = strategy["recommended_model"]
                    print(f"🎯 Agentic system recommends for answer processing: {recommended_model}")
                    
                    # Use the recommended model for processing
                    return _enhanced_process_student_pdf(filename, question_text, output_folder, recommended_model)
                else:
                    print("⚠️ Analysis failed, using fallback model")
                    return _enhanced_process_student_pdf(filename, question_text, output_folder, fallback_model)
                    
            finally:
                loop.close()
        else:
            print("🔧 Using enhanced processing method")
            return _enhanced_process_student_pdf(filename, question_text, output_folder, fallback_model)
            
    except Exception as e:
        print(f"❌ Error in agentic student processing: {e}")
        return _enhanced_process_student_pdf(filename, question_text, output_folder, fallback_model)

def _enhanced_process_student_pdf(filename: str, question_text: str, output_folder: str, model: str = "gemini"):
    """Enhanced processing with better question-answer mapping"""
    try:
        student_name = os.path.splitext(filename)[0]
        print(f"🔧 Enhanced processing: {student_name} with {model.upper()}")

        local_path = os.path.join("uploads/students_data", filename)
        if not os.path.exists(local_path):
            print(f"❌ File not found: {local_path}")
            return None
            
        print("📄 Converting student PDF to images...")
        image_pages = pdf_to_images(local_path)
        print(f"🖼️ Generated {len(image_pages)} pages")

        # Enhanced prompt with better question-answer mapping
        enhanced_prompt = f'''Create a comprehensive LaTeX document that maps student answers to exam questions.

REQUIRED OUTPUT: Complete LaTeX document starting with \\documentclass and ending with \\end{{document}}

STRUCTURE REQUIRED:
\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}

\\begin{{document}}
\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Questions and Student Responses}}

For each question below, show:
1. The complete question text
2. The student's complete answer

\\subsection*{{Question 1}}
\\textbf{{Question:}} [Question text from question paper]

\\textbf{{Student Answer:}}
\\begin{{quote}}
[Student's complete response]
\\end{{quote}}

\\subsection*{{Question 2}}
[Continue for all questions...]

\\end{{document}}

EXTRACTION REQUIREMENTS:
- Extract ALL student handwriting and marks
- Include mathematical calculations and diagrams
- Map answers to questions using question numbers when possible
- If mapping unclear, extract all content sequentially
- Describe diagrams as "Student drew: [description]"
- Do NOT correct answers - extract exactly as written
- ENSURE the output is COMPLETE and well-formed

QUESTION PAPER CONTENT:
{question_text}

STUDENT ANSWER SHEET:
Now examine the answer sheet images and create the complete LaTeX document.'''

        print(f"🤖 Extracting answers with enhanced mapping using {model.upper()}...")
        if model == "gemini":
            latex_output = gemini_extract_answer_latex(image_pages, question_text, enhanced_prompt)
        else:
            latex_output = gpt4o_extract_answer_latex(image_pages, question_text, enhanced_prompt)

        print(f"📝 Raw AI output preview: {latex_output[:300] if latex_output else 'No output'}...")
        
        # Enhanced cleaning and validation
        latex_output = enhanced_clean_latex_output(latex_output, question_text, student_name)
        
        tex_path = os.path.join(output_folder, f"{student_name}_answers.tex")
        print(f"💾 Writing enhanced LaTeX to: {tex_path}")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_output)

        print("🔨 Compiling LaTeX to PDF...")
        compile_command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory", output_folder,
            tex_path
        ]

        result = subprocess.run(compile_command, capture_output=True, text=True)
        
        # Try compilation twice (common LaTeX practice for references)
        if result.returncode == 0:
            subprocess.run(compile_command, capture_output=True, text=True)
        
        pdf_path = os.path.join(output_folder, f"{student_name}_answers.pdf")
        
        if not os.path.exists(pdf_path):
            print(f"❌ LaTeX compile error: {result.stderr}")
            print(f"📋 LaTeX stdout: {result.stdout}")
            
            # Create enhanced fallback PDF
            error_latex = create_enhanced_fallback_latex(
                f"LaTeX compilation failed.\n\nError: {result.stderr}\n\nGenerated LaTeX:\n{latex_output[:1000]}",
                question_text,
                student_name
            )
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(error_latex)
            
            # Try compiling the fallback
            fallback_result = subprocess.run(compile_command, capture_output=True, text=True)
            if fallback_result.returncode != 0 or not os.path.exists(pdf_path):
                print("❌ Even enhanced fallback compilation failed")
                return None
            
        print(f"✅ PDF generated for {student_name}")

        # Cleanup temporary files
        base_name = f"{student_name}_answers"
        cleanup_extensions = [".aux", ".log", ".tex", ".fdb_latexmk", ".fls", ".synctex.gz", ".out", ".toc"]
        
        for ext in cleanup_extensions:
            cleanup_file = os.path.join(output_folder, f"{base_name}{ext}")
            try:
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)
            except Exception as e:
                print(f"⚠️ Could not remove {cleanup_file}: {e}")

        return f"{student_name}_answers.pdf"
        
    except Exception as e:
        print(f"❌ Error in _enhanced_process_student_pdf: {e}")
        print(traceback.format_exc())
        return None

def enhanced_clean_latex_output(latex_text: str, question_text: str, student_name: str) -> str:
    """Enhanced LaTeX cleaning and validation with question integration"""
    if not latex_text or not latex_text.strip():
        return create_enhanced_fallback_latex("No content generated by AI model", question_text, student_name)
    
    # Remove markdown code blocks
    if "```latex" in latex_text:
        latex_text = latex_text.split("```latex")[1].split("```")[0]
    elif "```" in latex_text:
        parts = latex_text.split("```")
        if len(parts) >= 3:
            latex_text = parts[1]
    
    latex_text = latex_text.strip()
    
    # Check if it starts with proper LaTeX
    if not latex_text.startswith("\\documentclass"):
        # Try to find documentclass in the text
        match = re.search(r'\\documentclass.*?\\begin\{document\}.*?\\end\{document\}', latex_text, re.DOTALL)
        if match:
            latex_text = match.group(0)
        else:
            print("⚠️ Invalid LaTeX output detected, creating enhanced fallback document")
            return create_enhanced_fallback_latex(latex_text[:1000] + "..." if len(latex_text) > 1000 else latex_text, question_text, student_name)
    
    # Validate basic LaTeX structure
    required_elements = ["\\begin{document}", "\\end{document}"]
    for element in required_elements:
        if element not in latex_text:
            print(f"⚠️ Missing {element}, creating enhanced fallback")
            return create_enhanced_fallback_latex(latex_text[:1000] + "..." if len(latex_text) > 1000 else latex_text, question_text, student_name)
    
    # Check if questions are properly included
    if "Question" not in latex_text and "question" not in latex_text:
        print("⚠️ No questions found in output, enhancing with question content")
        return enhance_latex_with_questions(latex_text, question_text, student_name)
    
    return latex_text

def enhance_latex_with_questions(latex_content: str, question_text: str, student_name: str) -> str:
    """Enhance LaTeX output by properly integrating questions"""
    try:
        # Extract the body content between \begin{document} and \end{document}
        body_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        if body_match:
            body_content = body_match.group(1)
        else:
            body_content = latex_content
        
        # Create enhanced document with questions and answers
        enhanced_doc = f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\setlength{{\\parskip}}{{6pt}}

\\begin{{document}}

\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Exam Questions and Student Responses}}

\\subsection*{{Question Paper Content}}
\\begin{{quote}}
{question_text}
\\end{{quote}}

\\subsection*{{Student Response Content}}
\\begin{{quote}}
{body_content.strip()}
\\end{{quote}}

\\section*{{Processing Notes}}
The student's responses have been extracted from the answer sheet. The AI system attempted to map answers to specific questions, but manual review may be needed for complete accuracy.

\\end{{document}}"""
        
        return enhanced_doc
        
    except Exception as e:
        print(f"❌ Error enhancing LaTeX: {e}")
        return create_enhanced_fallback_latex("Error enhancing document", question_text, student_name)

def create_enhanced_fallback_latex(content: str, question_text: str, student_name: str) -> str:
    """Create an enhanced fallback LaTeX document with both questions and extracted content"""
    return f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry, enumitem}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\setlength{{\\parskip}}{{6pt}}

\\begin{{document}}

\\title{{Student Answer Sheet Analysis - {student_name}}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Processing Status}}
The AI model encountered difficulties generating a complete analysis. Below is the available content extracted from the documents.

\\section*{{Question Paper Content}}
\\begin{{quote}}
{question_text if question_text else "Question content not available"}
\\end{{quote}}

\\section*{{Extracted Student Work}}
\\begin{{quote}}
{content}
\\end{{quote}}

\\section*{{Recommendations}}
\\begin{{itemize}}
\\item Check image quality of the answer sheet
\\item Ensure handwriting is clear and legible  
\\item Try processing again with different AI model
\\item Consider manual review of the original documents
\\end{{itemize}}

\\section*{{Technical Details}}
\\begin{{itemize}}
\\item Student: {student_name}
\\item Processing Date: \\today
\\item AI Model: Agentic system with automatic selection
\\item Status: Partial extraction completed
\\end{{itemize}}

\\end{{document}}"""

def clean_latex_output(latex_text):
    """Clean and validate LaTeX output - legacy function maintained for compatibility"""
    if not latex_text or not latex_text.strip():
        return create_fallback_latex("No content generated by AI model")
    
    # Remove markdown code blocks
    if "```latex" in latex_text:
        latex_text = latex_text.split("```latex")[1].split("```")[0]
    elif "```" in latex_text:
        parts = latex_text.split("```")
        if len(parts) >= 3:
            latex_text = parts[1]
    
    latex_text = latex_text.strip()
    
    # Check if it starts with proper LaTeX
    if not latex_text.startswith("\\documentclass"):
        # Try to find documentclass in the text
        match = re.search(r'\\documentclass.*?\\begin\{document\}.*?\\end\{document\}', latex_text, re.DOTALL)
        if match:
            latex_text = match.group(0)
        else:
            print("WARNING: Invalid LaTeX output detected, creating fallback document")
            return create_fallback_latex(latex_text[:500] + "..." if len(latex_text) > 500 else latex_text)
    
    # Validate basic LaTeX structure
    if not ("\\begin{document}" in latex_text and "\\end{document}" in latex_text):
        print("WARNING: Missing document structure, creating fallback")
        return create_fallback_latex(latex_text[:500] + "..." if len(latex_text) > 500 else latex_text)
    
    return latex_text

def create_fallback_latex(content="Error processing content"):
    """Create a fallback LaTeX document - legacy function maintained for compatibility"""
    return f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\begin{{document}}
\\title{{Answer Sheet Processing Error}}
\\maketitle

\\section*{{Processing Error}}
The AI model did not generate valid LaTeX output. Raw content received:

\\begin{{verbatim}}
{content}
\\end{{verbatim}}

Please try processing again or check the input files.

\\end{{document}}"""

# New agentic processing function for direct use
async def process_exam_documents_agentic(question_pdf: str, answer_pdf: str, output_folder: str, fallback_model: str = "gemini"):
    """
    Direct agentic processing function for advanced use cases
    Returns detailed workflow information
    """
    if not AGENTIC_AVAILABLE:
        return {
            "success": False,
            "error": "Agentic system not available",
            "fallback_used": True
        }
    
    try:
        return await orchestrator.process_exam_documents(question_pdf, answer_pdf, output_folder, fallback_model)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_used": True
        }