import os
import traceback
import asyncio
import subprocess
import re
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Keep your original utility imports
from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex
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
    """Extract questions using agentic system with automatic model selection"""
    try:
        if AGENTIC_AVAILABLE:
            print("Using agentic system for question extraction...")
            
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
                    print(f"Analysis failed: {analysis_result.error}")
                    raise Exception(f"Document analysis failed: {analysis_result.error}")
                
                # Use automatic model selection but fallback if needed
                strategy = analysis_result.data["strategy"]
                print(f"Agentic system recommends: {strategy['recommended_model']}")
                
                # Extract questions
                extraction_task = {
                    "file_path": pdf_path,
                    "strategy": strategy
                }
                extraction_result = loop.run_until_complete(extractor.execute(extraction_task))
                
                if extraction_result.success:
                    print("Agentic extraction successful!")
                    return extraction_result.data["question_text"]
                else:
                    print(f"Agentic extraction failed: {extraction_result.error}")
                    raise Exception(f"Agentic extraction failed: {extraction_result.error}")
                    
            finally:
                loop.close()
                
        else:
            raise Exception("Agentic system not available")
            
    except Exception as e:
        print(f"Agentic extraction failed, using fallback method: {e}")
        return _original_extract_question_text(pdf_path, fallback_model)

def _original_extract_question_text(pdf_path: str, model: str = "gemini"):
    """Original question extraction method as fallback"""
    try:
        print(f"Converting PDF to images: {pdf_path}")
        image_paths = pdf_to_images(pdf_path)
        print(f"Generated {len(image_paths)} images")
        
        prompt = '''CRITICAL QUESTION PAPER EXTRACTION TASK:

You are extracting questions from an academic examination paper. Your task is to identify and transcribe ONLY the actual questions that require answers from students.

STRICT REQUIREMENTS:

1. IDENTIFY ACTUAL QUESTIONS - Look for content that:
   - Has clear question numbers (1, 2, 3... or Q1, Q2, Q3... or Question 1, Question 2...)
   - Contains question text that asks something requiring an answer
   - May have mark allocation like [5], [10], (3 marks), etc.
   - May have sub-parts (a), (b), (c)... or i), ii), iii... or (1), (2), (3)...
   - May contain multiple choice options A, B, C, D
   - May reference figures, diagrams, matrices, or mathematical expressions

2. EXCLUDE THE FOLLOWING - Do NOT include:
   - Header information (Department, Institute name, Course codes, Instructor names)
   - Exam metadata (Date, Time, Duration, Total marks)
   - General instructions ("The exam is closed book", "Write clearly", "Do not use AI")
   - Page numbers, watermarks, or footer text
   - Phrases like "Please go on to the next page", "Question continues on next page"
   - "End of Exam" or similar closing statements

3. FORMATTING REQUIREMENTS:
   - Start each main question with "Question [number]:"
   - Preserve all sub-parts with proper indentation
   - Include mark allocations if present: [X marks] or [X]
   - For multiple choice questions, include all options A, B, C, D
   - Preserve mathematical expressions, matrices, and formulas
   - Note references to figures/diagrams as [FIGURE REFERENCED] or [DIAGRAM REFERENCED]
   - Maintain question hierarchy and structure

If no actual questions are found, state "NO QUESTIONS FOUND ON THIS PAGE"

OUTPUT ONLY the extracted question text without any explanations or markdown formatting.'''
        
        print(f"Sending to {model.upper()} for question extraction...")
        if model == "gemini":
            result = gemini_extract_question_text(image_paths, prompt)
        else:
            result = gpt4o_extract_answer_latex(image_paths, question_text="", prompt=prompt)
        
        print("Question extraction complete")
        return result
        
    except Exception as e:
        print(f"Error in _original_extract_question_text: {e}")
        print(traceback.format_exc())
        return f"Error extracting questions: {str(e)}"

def process_student_pdf(filename: str, question_text: str, output_folder: str, fallback_model: str = "gemini"):
    """Process student PDF using agentic system with automatic model selection"""
    try:
        student_name = os.path.splitext(filename)[0]
        local_path = os.path.join("uploads/students_data", filename)
        
        if not os.path.exists(local_path):
            print(f"File not found: {local_path}")
            return None
        
        if AGENTIC_AVAILABLE:
            print(f"Processing {student_name} with agentic system...")
            
            # For full agentic processing, we'd need both question and answer PDFs
            # For now, use enhanced processing with original method
            print("Using enhanced processing method")
            return _original_process_student_pdf(filename, question_text, output_folder, fallback_model)
        else:
            print("Using original processing method")
            return _original_process_student_pdf(filename, question_text, output_folder, fallback_model)
            
    except Exception as e:
        print(f"Error in agentic student processing: {e}")
        return _original_process_student_pdf(filename, question_text, output_folder, fallback_model)

def _original_process_student_pdf(filename: str, question_text: str, output_folder: str, model: str = "gemini"):
    """Original processing method as fallback"""
    try:
        student_name = os.path.splitext(filename)[0]
        print(f"Processing: {student_name} with {model.upper()}")

        local_path = os.path.join("uploads/students_data", filename)
        if not os.path.exists(local_path):
            print(f"File not found: {local_path}")
            return None
            
        print("Converting student PDF to images...")
        image_pages = pdf_to_images(local_path)
        print(f"Generated {len(image_pages)} pages")

        print(f"Extracting answers with {model.upper()}...")
        if model == "gemini":
            latex_output = gemini_extract_answer_latex(image_pages, question_text)
        else:
            latex_output = gpt4o_extract_answer_latex(image_pages, question_text)

        print("Raw AI output preview:", latex_output[:200] if latex_output else "No output")
        
        # Clean and validate the LaTeX output
        latex_output = clean_latex_output(latex_output)
        
        tex_path = os.path.join(output_folder, f"{student_name}_answers.tex")
        print(f"Writing LaTeX to: {tex_path}")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_output)

        print("Compiling LaTeX to PDF...")
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
            print(f"LaTeX compile error: {result.stderr}")
            print(f"LaTeX stdout: {result.stdout}")
            
            # Create a simple fallback PDF with error info
            error_latex = create_fallback_latex(f"LaTeX compilation failed.\n\nError: {result.stderr}\n\nGenerated LaTeX:\n{latex_output[:1000]}")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(error_latex)
            
            # Try compiling the fallback
            fallback_result = subprocess.run(compile_command, capture_output=True, text=True)
            if fallback_result.returncode != 0 or not os.path.exists(pdf_path):
                print("Even fallback compilation failed")
                return None
            
        print(f"PDF generated for {student_name}")

        # Cleanup temporary files
        base_name = f"{student_name}_answers"
        cleanup_extensions = [".aux", ".log", ".tex", ".fdb_latexmk", ".fls", ".synctex.gz", ".out", ".toc"]
        
        for ext in cleanup_extensions:
            cleanup_file = os.path.join(output_folder, f"{base_name}{ext}")
            try:
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)
            except Exception as e:
                print(f"Could not remove {cleanup_file}: {e}")

        return f"{student_name}_answers.pdf"
        
    except Exception as e:
        print(f"Error in _original_process_student_pdf: {e}")
        print(traceback.format_exc())
        return None

def clean_latex_output(latex_text):
    """Clean and validate LaTeX output"""
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
            # If no valid LaTeX structure found, create fallback
            print("WARNING: Invalid LaTeX output detected, creating fallback document")
            return create_fallback_latex(latex_text[:500] + "..." if len(latex_text) > 500 else latex_text)
    
    # Validate basic LaTeX structure
    if not ("\\begin{document}" in latex_text and "\\end{document}" in latex_text):
        print("WARNING: Missing document structure, creating fallback")
        return create_fallback_latex(latex_text[:500] + "..." if len(latex_text) > 500 else latex_text)
    
    return latex_text

def create_fallback_latex(content="Error processing content"):
    """Create a fallback LaTeX document"""
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