# agents/answer_processor.py - Debug Enhanced Version
from typing import Dict, List, Any
from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex
from utils.ocr_gemini import gemini_extract_answer_latex
from .base_agent import BaseAgent, AgentResult

class AnswerProcessorAgent(BaseAgent):
    def __init__(self):
        super().__init__("AnswerProcessor", ["openai_vision", "gemini_vision"])
        self.answer_prompt = """Create a comprehensive LaTeX document mapping student answers to questions.

CRITICAL: Generate a COMPLETE document. Do not truncate or abbreviate.

REQUIRED STRUCTURE:
\\documentclass[12pt]{article}
\\usepackage{amsmath, amssymb, geometry, enumitem}
\\usepackage[utf8]{inputenc}
\\geometry{margin=1in}

\\begin{document}
\\title{Student Answer Sheet Analysis}
\\author{Automated Processing System}
\\date{\\today}
\\maketitle

\\section*{Questions and Student Responses}

\\subsection*{Question 1}
\\textbf{Question:} [Question text]
\\textbf{Student Answer:}
\\begin{quote}
[Student response]
\\end{quote}

\\subsection*{Question 2}
[Continue for all questions...]

\\end{document}

EXTRACTION RULES:
- Extract ALL student handwriting
- Map to questions when possible
- Include calculations, diagrams
- Generate COMPLETE document
- End with \\end{document}

CRITICAL: Ensure the output is COMPLETE and well-formed."""

    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            question_text = task["question_text"]
            strategy = task["strategy"]
            
            print(f"DEBUG: Processing answer sheet: {file_path}")
            print(f"DEBUG: Question text length: {len(question_text) if question_text else 0}")
            
            # Convert PDF to images
            image_paths = pdf_to_images(file_path)
            print(f"DEBUG: Generated {len(image_paths)} images from answer sheet")
            
            # Choose model based on strategy
            model = strategy["recommended_model"]
            print(f"DEBUG: Using model: {model}")
            
            # Create comprehensive prompt with question context
            full_prompt = self._create_debug_prompt(question_text)
            print(f"DEBUG: Prompt length: {len(full_prompt)}")
            
            # Process answers using chosen model
            latex_output = self._process_answers_debug(image_paths, question_text, model, full_prompt)
            print(f"DEBUG: Raw output length: {len(latex_output) if latex_output else 0}")
            
            if latex_output:
                print(f"DEBUG: Output preview: {latex_output[:500]}...")
                print(f"DEBUG: Output ending: ...{latex_output[-200:]}")
            
            # Enhanced validation
            validation = self._enhanced_validate_latex(latex_output)
            print(f"DEBUG: Validation result: {validation}")
            
            # Retry with different approach if validation fails
            if not validation["is_valid"]:
                print("DEBUG: First attempt failed, trying simplified approach...")
                simplified_prompt = self._create_simplified_prompt(question_text)
                latex_output = self._process_answers_debug(image_paths, question_text, model, simplified_prompt)
                validation = self._enhanced_validate_latex(latex_output)
                
                if not validation["is_valid"]:
                    print("DEBUG: Second attempt failed, creating structured fallback...")
                    latex_output = self._create_structured_fallback(latex_output, question_text)
                    validation = {"is_valid": True, "confidence": 0.6, "issues": ["Used structured fallback"]}
            
            return AgentResult(
                success=validation["is_valid"],
                data={
                    "latex_output": latex_output,
                    "model_used": model,
                    "validation": validation,
                    "image_paths": image_paths
                },
                confidence=validation["confidence"]
            )
            
        except Exception as e:
            print(f"DEBUG: Exception in answer processor: {e}")
            import traceback
            traceback.print_exc()
            return AgentResult(success=False, error=str(e))
    
    def _create_debug_prompt(self, question_text: str) -> str:
        return f"""Generate a complete LaTeX document. CRITICAL: Do not truncate the output.

COMPLETE LATEX TEMPLATE:
\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}

\\begin{{document}}
\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing}}
\\date{{\\today}}
\\maketitle

\\section*{{Questions and Answers}}

For each question below, show the question text followed by the student's answer.

[EXTRACT STUDENT RESPONSES AND MAP TO QUESTIONS]

\\end{{document}}

QUESTION PAPER:
{question_text if question_text else "No questions provided"}

STUDENT ANSWER SHEET: Extract all student work and create the complete LaTeX document above."""
    
    def _create_simplified_prompt(self, question_text: str) -> str:
        return f"""Create a simple but complete LaTeX document:

\\documentclass{{article}}
\\usepackage{{amsmath}}
\\begin{{document}}
\\title{{Student Answers}}
\\maketitle

\\section{{Student Work}}
[Extract all student handwriting and work]

\\section{{Questions}}
{question_text if question_text else "Questions not available"}

\\end{{document}}

Extract ALL student work from the images and create this complete document."""
    
    def _process_answers_debug(self, image_paths: List[str], question_text: str, model: str, prompt: str) -> str:
        try:
            print(f"DEBUG: Processing with {model}, {len(image_paths)} images")
            
            if model == "gemini":
                result = gemini_extract_answer_latex(image_paths, question_text, prompt)
            else:
                result = gpt4o_extract_answer_latex(image_paths, question_text, prompt)
            
            print(f"DEBUG: Model returned {len(result) if result else 0} characters")
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in model processing: {e}")
            return f"Error in processing: {str(e)}"
    
    def _enhanced_validate_latex(self, latex_output: str) -> Dict:
        validation = {
            "is_valid": True,
            "should_retry": False,
            "confidence": 0.9,
            "issues": []
        }
        
        if not latex_output or len(latex_output.strip()) < 100:
            validation["is_valid"] = False
            validation["should_retry"] = True
            validation["confidence"] = 0.1
            validation["issues"].append("Output too short")
            return validation
        
        # Check for LaTeX structure
        required_elements = [
            ("\\documentclass", "Missing \\documentclass"),
            ("\\begin{document}", "Missing \\begin{document}"),
            ("\\end{document}", "Missing \\end{document}")
        ]
        
        for element, error_msg in required_elements:
            if element not in latex_output:
                validation["is_valid"] = False
                validation["confidence"] = 0.3
                validation["issues"].append(error_msg)
        
        # Check for content between begin and end document
        import re
        doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_output, re.DOTALL)
        if doc_match:
            content = doc_match.group(1).strip()
            if len(content) < 50:
                validation["is_valid"] = False
                validation["confidence"] = 0.4
                validation["issues"].append("Insufficient content in document")
        else:
            validation["is_valid"] = False
            validation["confidence"] = 0.2
            validation["issues"].append("Cannot find document content")
        
        return validation
    
    def _create_structured_fallback(self, original_output: str, question_text: str) -> str:
        """Create a well-structured fallback document"""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath, amssymb, geometry}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}

\\begin{{document}}

\\title{{Student Answer Sheet Analysis}}
\\author{{Automated Processing System}}
\\date{{\\today}}
\\maketitle

\\section*{{Processing Status}}
The system encountered difficulties generating a complete analysis. Available content is shown below.

\\section*{{Question Paper Content}}
\\begin{{quote}}
{question_text if question_text else "Question content not available"}
\\end{{quote}}

\\section*{{Extracted Content}}
\\begin{{quote}}
{original_output[:1000] if original_output else "No content extracted successfully"}
\\end{{quote}}

\\section*{{Technical Information}}
\\begin{{itemize}}
\\item Processing method: Enhanced agentic system with fallback
\\item Issue: LaTeX generation validation failed
\\item Status: Partial content extracted
\\item Recommendation: Check source document quality and retry
\\end{{itemize}}

\\end{{document}}"""