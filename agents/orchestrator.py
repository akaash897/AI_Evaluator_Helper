# agents/orchestrator.py
import os
import shutil
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent, AgentResult
from .document_analyzer import DocumentAnalyzerAgent
from .question_extractor import QuestionExtractorAgent
from .answer_processor import AnswerProcessorAgent
from .latex_compiler import LatexCompilerAgent

class ExamProcessingOrchestrator:
    def __init__(self):
        self.agents = {
            "analyzer": DocumentAnalyzerAgent(),
            "question_extractor": QuestionExtractorAgent(),
            "answer_processor": AnswerProcessorAgent(),
            "latex_compiler": LatexCompilerAgent()
        }
        self.workflow_state = {}
        self.max_retries = 2
    
    async def process_exam_documents(self, question_pdf: str, answer_pdf: str, output_folder: str, selected_model: str = "gemini") -> Dict[str, Any]:
        """Main orchestration method that coordinates all agents"""
        
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workflow_state = {
            "id": workflow_id,
            "question_pdf": question_pdf,
            "answer_pdf": answer_pdf,
            "output_folder": output_folder,
            "selected_model": selected_model,
            "steps": [],
            "current_step": 0,
            "errors": [],
            "retry_count": 0
        }
        
        try:
            # Step 1: Analyze question document
            print("Step 1: Analyzing question document...")
            q_analysis_result = await self._execute_agent(
                "analyzer",
                {
                    "file_path": question_pdf,
                    "file_type": "question_paper"
                }
            )
            
            if not q_analysis_result.success:
                return self._create_error_response("Question document analysis failed", q_analysis_result.error)
            
            # Step 2: Analyze answer document  
            print("Step 2: Analyzing answer document...")
            a_analysis_result = await self._execute_agent(
                "analyzer", 
                {
                    "file_path": answer_pdf,
                    "file_type": "answer_sheet"
                }
            )
            
            if not a_analysis_result.success:
                return self._create_error_response("Answer document analysis failed", a_analysis_result.error)
            
            # Step 3: Extract questions
            print("Step 3: Extracting questions...")
            # Use analyzer's recommendation or user preference
            q_strategy = q_analysis_result.data["strategy"].copy()
            if selected_model != "auto":
                q_strategy["recommended_model"] = selected_model
            # else: keep analyzer's recommendation
            
            question_result = await self._execute_agent(
                "question_extractor",
                {
                    "file_path": question_pdf,
                    "strategy": q_strategy
                }
            )
            
            if not question_result.success:
                return self._create_error_response("Question extraction failed", question_result.error)
            
            # Step 4: Process answers
            print("Step 4: Processing answer sheet...")
            # Use analyzer's recommendation or user preference
            a_strategy = a_analysis_result.data["strategy"].copy()
            if selected_model != "auto":
                a_strategy["recommended_model"] = selected_model
            # else: keep analyzer's recommendation
            
            answer_result = await self._execute_agent(
                "answer_processor",
                {
                    "file_path": answer_pdf,
                    "question_text": question_result.data["question_text"],
                    "strategy": a_strategy
                }
            )
            
            if not answer_result.success:
                return self._create_error_response("Answer processing failed", answer_result.error)
            
            # Step 5: Compile LaTeX
            print("Step 5: Compiling LaTeX...")
            student_name = os.path.splitext(os.path.basename(answer_pdf))[0]
            
            compile_result = await self._execute_agent(
                "latex_compiler",
                {
                    "latex_content": answer_result.data["latex_output"],
                    "output_folder": output_folder,
                    "filename": f"{student_name}_answers"
                }
            )
            
            if not compile_result.success:
                return self._create_error_response("LaTeX compilation failed", compile_result.error)
            
            # Cleanup temporary files
            self._cleanup_temp_files(output_folder, f"{student_name}_answers")
            
            # Success!
            return {
                "success": True,
                "pdf_filename": compile_result.data["filename"],
                "pdf_path": compile_result.data["pdf_path"],
                "workflow_state": self.workflow_state,
                "model_used": selected_model
            }
            
        except Exception as e:
            return self._create_error_response("Unexpected error in orchestration", str(e))
    
    async def _execute_agent(self, agent_name: str, task: Dict[str, Any]) -> AgentResult:
        """Execute an agent with retry logic"""
        agent = self.agents[agent_name]
        
        for attempt in range(self.max_retries + 1):
            try:
                print(f"  Executing {agent_name} (attempt {attempt + 1})...")
                result = await agent.execute(task)
                
                # Log the execution
                agent.log_execution(task, result)
                self._log_workflow_step(agent_name, result, attempt + 1)
                
                if result.success:
                    return result
                elif attempt < self.max_retries:
                    print(f"  {agent_name} failed, retrying... Error: {result.error}")
                    # Modify task for retry (could implement specific retry strategies)
                    task = self._modify_task_for_retry(agent_name, task, result.error)
                else:
                    print(f"  {agent_name} failed after {self.max_retries + 1} attempts")
                    return result
                    
            except Exception as e:
                print(f"  Exception in {agent_name}: {str(e)}")
                if attempt >= self.max_retries:
                    return AgentResult(success=False, error=str(e))
        
        return AgentResult(success=False, error="Max retries exceeded")
    
    def _modify_task_for_retry(self, agent_name: str, task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Modify task parameters for retry attempts"""
        modified_task = task.copy()
        
        if agent_name == "question_extractor" and "strategy" in task:
            # Try different model on retry
            current_model = task["strategy"]["recommended_model"]
            modified_task["strategy"]["recommended_model"] = "gemini" if current_model == "openai" else "openai"
        
        elif agent_name == "answer_processor" and "strategy" in task:
            # Try different model on retry
            current_model = task["strategy"]["recommended_model"]
            modified_task["strategy"]["recommended_model"] = "gemini" if current_model == "openai" else "openai"
        
        return modified_task
    
    def _log_workflow_step(self, agent_name: str, result: AgentResult, attempt: int):
        """Log workflow step for debugging and monitoring"""
        step_info = {
            "agent": agent_name,
            "attempt": attempt,
            "success": result.success,
            "confidence": result.confidence,
            "error": result.error,
            "timestamp": result.timestamp
        }
        self.workflow_state["steps"].append(step_info)
    
    def _create_error_response(self, message: str, error: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": message,
            "details": error,
            "workflow_state": self.workflow_state
        }
    
    def _cleanup_temp_files(self, output_folder: str, base_name: str):
        """Clean up temporary LaTeX files"""
        cleanup_extensions = [".aux", ".log", ".tex", ".fdb_latexmk", ".fls", ".synctex.gz", ".out", ".toc"]
        
        for ext in cleanup_extensions:
            cleanup_file = os.path.join(output_folder, f"{base_name}{ext}")
            try:
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)
            except Exception as e:
                print(f"Could not remove {cleanup_file}: {e}")