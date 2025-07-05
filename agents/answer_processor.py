# agents/answer_processor.py
from typing import Dict, List, Any
from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex
from utils.ocr_gemini import gemini_extract_answer_latex
from .base_agent import BaseAgent, AgentResult

class AnswerProcessorAgent(BaseAgent):
    def __init__(self):
        super().__init__("AnswerProcessor", ["openai_vision", "gemini_vision"])
        self.answer_prompt = (
            "You are an expert assistant for exam paper processing. Your job is to extract BOTH the questions and the answers, and map each question to its corresponding answer.\n"
            "\nIMPORTANT: You MUST output a complete, valid LaTeX document starting with \\documentclass and ending with \\end{document}.\n"
            "\nSTRICT INSTRUCTIONS:\n"
            "1. For each question in the question paper, extract the full question text and write it in the output.\n"
            "2. For each answer in the answer sheet, match it to the correct question number and place it directly after the corresponding question.\n"
            "3. If a question is not answered, write 'Not answered' under that question.\n"
            "4. Do NOT skip or summarize any part of the student's answer. Write everything exactly as written, including diagrams (describe them if present), points, arrows, and all formatting.\n"
            "5. For numerical or math problems, do NOT correct or reason about the answer. Just extract as written.\n"
            "6. Use proper LaTeX formatting. Include all required packages for math, tables, and diagrams.\n"
            "7. Output ONLY a complete LaTeX document, starting with \\documentclass and ending with \\end{document}. Do NOT include any markdown formatting, explanations, or code blocks.\n"
            "8. The output should have, for each question:\n"
            "   - The question number and full question text (from the question paper)\n"
            "   - The answer (from the answer sheet) matched to that question\n"
            "\nEXAMPLE OUTPUT FORMAT:\n"
            "\\documentclass{article}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{margin=1in}\n"
            "\\begin{document}\n"
            "\\title{Student Answer Sheet}\n"
            "\\maketitle\n"
            "\\section*{Question 1}\n"
            "What is the capital of France?\n"
            "\\textbf{Answer:}\n"
            "Paris\n"
            "\\section*{Question 2}\n"
            "Solve: $x^2 - 4 = 0$\n"
            "\\textbf{Answer:}\n"
            "$x^2 - 4 = 0$ implies $x^2 = 4$ therefore $x = \\pm 2$\n"
            "\\end{document}\n"
            "\nCRITICAL: Your response must start with \\documentclass and end with \\end{document}. Do not include any other text before or after the LaTeX document.\n"
        )
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            question_text = task["question_text"]
            strategy = task["strategy"]
            
            # Convert PDF to images
            image_paths = pdf_to_images(file_path)
            
            # Choose model based on strategy
            model = strategy["recommended_model"]
            
            # Process answers using chosen model
            latex_output = self._process_answers(image_paths, question_text, model)
            
            # Validate LaTeX output
            validation = self._validate_latex(latex_output)
            
            # Retry with different model if validation fails
            if not validation["is_valid"] and validation["should_retry"]:
                fallback_model = "gemini" if model == "openai" else "openai"
                latex_output = self._process_answers(image_paths, question_text, fallback_model)
                validation = self._validate_latex(latex_output)
            
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
            return AgentResult(success=False, error=str(e))
    
    def _process_answers(self, image_paths: List[str], question_text: str, model: str) -> str:
        full_prompt = self.answer_prompt + f"\nQUESTION PAPER:\n{question_text}\nANSWER SHEET IMAGES:"
        
        if model == "gemini":
            return gemini_extract_answer_latex(image_paths, question_text, full_prompt)
        else:
            return gpt4o_extract_answer_latex(image_paths, question_text, full_prompt)
    
    def _validate_latex(self, latex_output: str) -> Dict:
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
            validation["issues"].append("LaTeX output too short")
            return validation
        
        # Check for LaTeX structure
        if not latex_output.strip().startswith("\\documentclass"):
            validation["is_valid"] = False
            validation["confidence"] = 0.3
            validation["issues"].append("Missing \\documentclass")
        
        if "\\begin{document}" not in latex_output:
            validation["is_valid"] = False
            validation["confidence"] = 0.3
            validation["issues"].append("Missing \\begin{document}")
        
        if "\\end{document}" not in latex_output:
            validation["is_valid"] = False
            validation["confidence"] = 0.3
            validation["issues"].append("Missing \\end{document}")
        
        return validation