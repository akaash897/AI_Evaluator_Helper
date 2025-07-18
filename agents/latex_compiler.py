# agents/latex_compiler.py
import subprocess
import os
import re
from typing import Dict, Any
from .base_agent import BaseAgent, AgentResult

class LatexCompilerAgent(BaseAgent):
    def __init__(self):
        super().__init__("LatexCompiler", ["pdflatex"])
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            latex_content = task["latex_content"]
            output_folder = task["output_folder"]
            filename = task["filename"]
            
            # Clean LaTeX content
            cleaned_latex = self._clean_latex_output(latex_content)
            
            # Write LaTeX file
            tex_path = os.path.join(output_folder, f"{filename}.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(cleaned_latex)
            
            # Compile LaTeX
            compilation_result = self._compile_latex(tex_path, output_folder)
            
            # If compilation fails, try to fix and retry
            if not compilation_result["success"]:
                fixed_latex = self._fix_latex_errors(cleaned_latex, compilation_result["errors"])
                
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(fixed_latex)
                
                compilation_result = self._compile_latex(tex_path, output_folder)
            
            pdf_path = os.path.join(output_folder, f"{filename}.pdf")
            
            return AgentResult(
                success=compilation_result["success"],
                data={
                    "pdf_path": pdf_path if compilation_result["success"] else None,
                    "tex_path": tex_path,
                    "compilation_log": compilation_result["log"],
                    "filename": f"{filename}.pdf" if compilation_result["success"] else None
                },
                confidence=1.0 if compilation_result["success"] else 0.0
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def _clean_latex_output(self, latex_text: str) -> str:
        """Clean and validate LaTeX output"""
        if not latex_text or not latex_text.strip():
            return self._create_fallback_latex("No content generated by AI model")
        
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
                return self._create_fallback_latex(latex_text[:500] + "..." if len(latex_text) > 500 else latex_text)
        
        # Validate basic LaTeX structure
        if not ("\\begin{document}" in latex_text and "\\end{document}" in latex_text):
            print("WARNING: Missing document structure, creating fallback")
            return self._create_fallback_latex(latex_text[:500] + "..." if len(latex_text) > 500 else latex_text)
        
        return latex_text
    
    def _create_fallback_latex(self, content="Error processing content") -> str:
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
    
    def _compile_latex(self, tex_path: str, output_folder: str) -> Dict:
        try:
            compile_command = [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory", output_folder,
                tex_path
            ]
            
            result = subprocess.run(compile_command, capture_output=True, text=True)
            
            # Try compilation twice (common LaTeX practice)
            if result.returncode == 0:
                subprocess.run(compile_command, capture_output=True, text=True)
            
            pdf_path = tex_path.replace(".tex", ".pdf")
            success = os.path.exists(pdf_path)
            
            return {
                "success": success,
                "log": result.stdout,
                "errors": result.stderr
            }
            
        except Exception as e:
            return {
                "success": False,
                "log": "",
                "errors": str(e)
            }
    
    def _fix_latex_errors(self, latex_content: str, errors: str) -> str:
        """Auto-fix common LaTeX errors"""
        fixed_content = latex_content
        
        # Fix common issues
        if "Undefined control sequence" in errors:
            # Add missing packages
            if "\\textbf" in fixed_content and "\\usepackage{textcomp}" not in fixed_content:
                fixed_content = fixed_content.replace("\\begin{document}", "\\usepackage{textcomp}\n\\begin{document}")
        
        if "Missing $ inserted" in errors:
            # This is harder to fix automatically, return fallback
            return self._create_fallback_latex("Math formatting error in original content")
        
        return fixed_content