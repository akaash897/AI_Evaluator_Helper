# agents/document_analyzer.py - Enhanced with research-based selection

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
            
            print(f"🔍 Analyzing document: {os.path.basename(file_path)}")
            print(f"📄 Document type: {file_type}")
            
            analysis = self._analyze_document(file_path)
            strategy = self._determine_processing_strategy_research_based(analysis, file_type)
            
            # Print selection reasoning
            self._print_selection_reasoning(strategy, analysis, file_type)
            
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
            
            # Enhanced analysis based on research factors
            analysis = {
                "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2),
                "image_dimensions": image.size,
                "estimated_pages": self._estimate_page_count(file_path),
                "image_quality": self._assess_image_quality(image),
                "complexity": self._assess_document_complexity(image),
                "text_density": self._estimate_text_density(image),
                "has_handwriting": True,  # Assume true for exam sheets
                "confidence": 0.85
            }
            
            print(f"📊 Document Analysis:")
            print(f"   • Pages: {analysis['estimated_pages']}")
            print(f"   • Size: {analysis['file_size_mb']} MB")
            print(f"   • Resolution: {analysis['image_dimensions'][0]}x{analysis['image_dimensions'][1]}")
            print(f"   • Quality: {analysis['image_quality']}")
            print(f"   • Complexity: {analysis['complexity']}")
            
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
        """Assess image quality based on resolution and clarity"""
        width, height = image.size
        total_pixels = width * height
        
        if total_pixels < 500000:  # Less than 0.5MP
            return "low"
        elif total_pixels > 4000000:  # More than 4MP
            return "high"
        else:
            return "medium"
    
    def _assess_document_complexity(self, image: Image.Image) -> str:
        """Estimate document complexity"""
        width, height = image.size
        
        # Simple heuristic: larger images often indicate more complex layouts
        if width > 2500 and height > 3000:
            return "high"    # Likely detailed exam paper or dense answer sheet
        elif width < 1200 or height < 1500:
            return "low"     # Simple document or poor scan
        else:
            return "medium"
    
    def _estimate_text_density(self, image: Image.Image) -> str:
        """Estimate how much text is in the document"""
        # This is a simplified heuristic - in practice you might use edge detection
        width, height = image.size
        
        # Convert to grayscale and estimate "busy-ness"
        try:
            gray = image.convert('L')
            # Simple estimation based on image size and type
            if width * height > 3000000:  # Large, likely dense document
                return "high"
            else:
                return "medium"
        except:
            return "medium"
    
    def _determine_processing_strategy_research_based(self, analysis: Dict, file_type: str) -> Dict:
        """Research-based model selection between OpenAI and Gemini"""
        
        strategy = {
            "recommended_model": "openai",  # Default
            "dpi_setting": 300,
            "preprocessing_needed": False,
            "retry_strategy": "fallback_model",
            "reasoning": []
        }
        
        # Research-based selection criteria
        
        # 1. DOCUMENT TYPE (Primary Factor)
        if file_type == "question_paper":
            # Research shows OpenAI better for structured/printed documents
            strategy["recommended_model"] = "openai"
            strategy["reasoning"].append("Question papers: OpenAI excels at structured text")
            
        elif file_type == "answer_sheet":
            # Research shows mixed results, but consider other factors
            if analysis.get("complexity") == "high":
                strategy["recommended_model"] = "openai"
                strategy["reasoning"].append("Complex handwriting: OpenAI better for detailed analysis")
            else:
                strategy["recommended_model"] = "gemini"
                strategy["reasoning"].append("Answer sheets: Gemini faster for handwriting")
        
        # 2. IMAGE QUALITY (Secondary Factor)
        if analysis.get("image_quality") == "low":
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append("Poor image quality: Gemini handles noise better")
            strategy["dpi_setting"] = 400  # Higher DPI for poor quality
            
        elif analysis.get("image_quality") == "high":
            strategy["recommended_model"] = "openai"
            strategy["reasoning"].append("High quality images: OpenAI maximizes detail extraction")
        
        # 3. DOCUMENT SIZE (Performance Factor)
        if analysis.get("estimated_pages", 1) > 10:
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append("Large document: Gemini faster and more cost-effective")
            
        elif analysis.get("estimated_pages", 1) > 5:
            if strategy["recommended_model"] == "openai":
                strategy["reasoning"].append("Medium size: OpenAI acceptable for up to 5-10 pages")
        
        # 4. COMPLEXITY (Detail Factor)
        if analysis.get("complexity") == "high" and analysis.get("text_density") == "high":
            strategy["recommended_model"] = "openai"
            strategy["reasoning"].append("High complexity + density: OpenAI's precision advantage")
            
        # 5. FILE SIZE (Practical Factor)
        if analysis.get("file_size_mb", 0) > 50:
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append("Large file size: Gemini handles big files better")
        
        # 6. COST OPTIMIZATION (Research shows Gemini more cost-efficient)
        if analysis.get("estimated_pages", 1) > 15:
            original_model = strategy["recommended_model"]
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append(f"Cost optimization: Switched from {original_model} to Gemini for large document")
        
        return strategy
    
    def _print_selection_reasoning(self, strategy: Dict, analysis: Dict, file_type: str):
        """Print detailed reasoning for model selection"""
        selected_model = strategy["recommended_model"]
        reasoning = strategy.get("reasoning", [])
        
        print(f"🤖 Model Selection: {selected_model.upper()}")
        print(f"🧠 Selection Reasoning:")
        
        if reasoning:
            for i, reason in enumerate(reasoning, 1):
                print(f"   {i}. {reason}")
        else:
            print(f"   • Default selection for {file_type}")
        
        # Research-based expectations
        if selected_model == "openai":
            print(f"📈 Expected Strengths:")
            print(f"   • Higher precision and accuracy")
            print(f"   • Better for structured/printed text")
            print(f"   • Superior mathematical expression handling")
            
        elif selected_model == "gemini":
            print(f"📈 Expected Strengths:")
            print(f"   • Faster processing speed")
            print(f"   • Better cost efficiency")
            print(f"   • Good performance on handwriting")
            print(f"   • Handles poor image quality well")