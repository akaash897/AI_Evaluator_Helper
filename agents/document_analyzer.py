# agents/document_analyzer.py
import os
from typing import Dict, List, Any
from pdf2image import convert_from_path
from PIL import Image
from .base_agent import BaseAgent, AgentResult

class DocumentAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__("DocumentAnalyzer", ["pdf_reader", "image_converter"])
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        try:
            file_path = task["file_path"]
            file_type = task.get("file_type", "unknown")
            
            analysis = self._analyze_document(file_path)
            strategy = self._determine_processing_strategy(analysis, file_type)
            
            return AgentResult(
                success=True,
                data={
                    "analysis": analysis,
                    "strategy": strategy,
                    "file_path": file_path,
                    "file_type": file_type
                },
                confidence=analysis.get("confidence", 0.9)
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def _analyze_document(self, file_path: str) -> Dict:
        try:
            # Convert first page to analyze document characteristics
            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=1)
            
            if not images:
                return {"confidence": 0.0, "error": "Could not convert PDF"}
            
            image = images[0]
            
            # Basic analysis
            analysis = {
                "file_size": os.path.getsize(file_path),
                "image_dimensions": image.size,
                "estimated_pages": self._estimate_page_count(file_path),
                "image_quality": self._assess_image_quality(image),
                "has_handwriting": True,  # Assume true for exam sheets
                "confidence": 0.85
            }
            
            return analysis
            
        except Exception as e:
            return {"confidence": 0.0, "error": str(e)}
    
    def _estimate_page_count(self, file_path: str) -> int:
        try:
            images = convert_from_path(file_path, dpi=72)
            return len(images)
        except:
            return 1
    
    def _assess_image_quality(self, image: Image.Image) -> str:
        width, height = image.size
        if width < 1000 or height < 1000:
            return "low"
        elif width > 2000 and height > 2000:
            return "high"
        else:
            return "medium"
    
    def _determine_processing_strategy(self, analysis: Dict, file_type: str) -> Dict:
        strategy = {
            "recommended_model": "openai",  # Default
            "dpi_setting": 300,
            "preprocessing_needed": False,
            "retry_strategy": "fallback_model"
        }
        
        # Model selection based on analysis
        if analysis.get("image_quality") == "low":
            strategy["recommended_model"] = "gemini"  # Better for poor quality
            strategy["dpi_setting"] = 400  # Higher DPI for poor quality
        elif analysis.get("estimated_pages", 1) > 10:
            strategy["recommended_model"] = "gemini"  # Faster for large docs
        
        # File type specific adjustments
        if file_type == "question_paper":
            strategy["extraction_focus"] = "structured_text"
        elif file_type == "answer_sheet":
            strategy["extraction_focus"] = "handwriting"
        
        return strategy