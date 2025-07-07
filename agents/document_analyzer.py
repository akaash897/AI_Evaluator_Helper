# agents/document_analyzer.py - Enhanced with research-based selection and multi-page support

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
            
            analysis = self._analyze_document_multipage(file_path)
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
    
    def _analyze_document_multipage(self, file_path: str) -> Dict:
        """Enhanced analysis with full multi-page support"""
        try:
            # Get total page count first
            total_pages = self._get_total_page_count(file_path)
            
            # Convert first few pages for quality analysis
            pages_to_analyze = min(3, total_pages)  # Analyze up to 3 pages for better assessment
            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=pages_to_analyze)
            
            if not images:
                return {"confidence": 0.0, "error": "Could not convert PDF", "total_pages": 0}
            
            # Analyze first page in detail
            first_page = images[0]
            
            # Enhanced analysis based on research factors
            analysis = {
                "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2),
                "image_dimensions": first_page.size,
                "total_pages": total_pages,
                "pages_analyzed": len(images),
                "image_quality": self._assess_image_quality(first_page),
                "complexity": self._assess_document_complexity_multipage(images),
                "text_density": self._estimate_text_density_multipage(images),
                "document_type_confidence": self._assess_document_type_confidence(images),
                "has_handwriting": True,  # Assume true for exam sheets
                "confidence": 0.85
            }
            
            print(f"📊 Multi-page Document Analysis:")
            print(f"   • Total Pages: {analysis['total_pages']}")
            print(f"   • Pages Analyzed: {analysis['pages_analyzed']}")
            print(f"   • File Size: {analysis['file_size_mb']} MB")
            print(f"   • Resolution: {analysis['image_dimensions'][0]}x{analysis['image_dimensions'][1]}")
            print(f"   • Quality: {analysis['image_quality']}")
            print(f"   • Complexity: {analysis['complexity']}")
            print(f"   • Text Density: {analysis['text_density']}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error in document analysis: {e}")
            return {"confidence": 0.0, "error": str(e), "total_pages": 1}
    
    def _get_total_page_count(self, file_path: str) -> int:
        """Get total page count efficiently"""
        try:
            # Use low DPI for quick page counting
            images = convert_from_path(file_path, dpi=72)
            return len(images)
        except Exception as e:
            print(f"⚠️ Could not count pages: {e}")
            return 1
    
    def _assess_document_complexity_multipage(self, images: List[Image.Image]) -> str:
        """Assess document complexity across multiple pages"""
        complexities = []
        
        for i, image in enumerate(images):
            width, height = image.size
            
            # Analyze each page
            if width > 2500 and height > 3000:
                complexities.append("high")
            elif width < 1200 or height < 1500:
                complexities.append("low")
            else:
                complexities.append("medium")
        
        # Determine overall complexity
        if "high" in complexities:
            return "high"
        elif all(c == "low" for c in complexities):
            return "low"
        else:
            return "medium"
    
    def _estimate_text_density_multipage(self, images: List[Image.Image]) -> str:
        """Estimate text density across multiple pages"""
        densities = []
        
        for image in images:
            width, height = image.size
            
            try:
                gray = image.convert('L')
                # Simple estimation based on image size and analysis
                if width * height > 3000000:
                    densities.append("high")
                elif width * height < 1000000:
                    densities.append("low")
                else:
                    densities.append("medium")
            except:
                densities.append("medium")
        
        # Determine overall density
        if "high" in densities:
            return "high"
        elif all(d == "low" for d in densities):
            return "low"
        else:
            return "medium"
    
    def _assess_document_type_confidence(self, images: List[Image.Image]) -> float:
        """Assess confidence in document type classification"""
        # Simple heuristic based on consistency across pages
        if len(images) > 1:
            # Multi-page documents are typically more structured
            return 0.9
        else:
            return 0.8
    
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
    
    def _determine_processing_strategy_research_based(self, analysis: Dict, file_type: str) -> Dict:
        """Research-based model selection with multi-page considerations"""
        
        strategy = {
            "recommended_model": "openai",  # Default
            "dpi_setting": 300,
            "preprocessing_needed": False,
            "retry_strategy": "fallback_model",
            "multi_page_strategy": "batch_process",  # New field
            "reasoning": []
        }
        
        total_pages = analysis.get("total_pages", 1)
        
        # 1. MULTI-PAGE SPECIFIC LOGIC (Primary Factor)
        if total_pages > 10:
            strategy["recommended_model"] = "gemini"
            strategy["multi_page_strategy"] = "page_by_page_fallback"
            strategy["reasoning"].append(f"Large document ({total_pages} pages): Gemini better for bulk processing")
            
        elif total_pages > 5:
            if file_type == "question_paper":
                strategy["recommended_model"] = "openai"
                strategy["reasoning"].append(f"Multi-page questions ({total_pages} pages): OpenAI better for structured content")
            else:
                strategy["recommended_model"] = "gemini"
                strategy["reasoning"].append(f"Multi-page answers ({total_pages} pages): Gemini faster for handwriting")
        
        # 2. DOCUMENT TYPE (Enhanced for multi-page)
        if file_type == "question_paper":
            if total_pages <= 3:
                strategy["recommended_model"] = "openai"
                strategy["reasoning"].append("Short question paper: OpenAI excels at structured text")
            else:
                # For longer question papers, consider other factors
                strategy["reasoning"].append("Long question paper: considering other factors...")
                
        elif file_type == "answer_sheet":
            if analysis.get("complexity") == "high" and total_pages <= 5:
                strategy["recommended_model"] = "openai"
                strategy["reasoning"].append("Complex handwriting (few pages): OpenAI better for detailed analysis")
            else:
                strategy["recommended_model"] = "gemini"
                strategy["reasoning"].append("Answer sheets: Gemini efficient for handwriting processing")
        
        # 3. IMAGE QUALITY (Secondary Factor)
        if analysis.get("image_quality") == "low":
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append("Poor image quality: Gemini handles noise better")
            strategy["dpi_setting"] = 400  # Higher DPI for poor quality
            
        elif analysis.get("image_quality") == "high" and total_pages <= 5:
            strategy["recommended_model"] = "openai"
            strategy["reasoning"].append("High quality images: OpenAI maximizes detail extraction")
        
        # 4. COMPLEXITY AND DENSITY (Detailed Factor)
        complexity = analysis.get("complexity", "medium")
        text_density = analysis.get("text_density", "medium")
        
        if complexity == "high" and text_density == "high" and total_pages <= 3:
            strategy["recommended_model"] = "openai"
            strategy["reasoning"].append("High complexity + density (short doc): OpenAI's precision advantage")
        elif complexity == "high" and total_pages > 5:
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append("High complexity + many pages: Gemini's efficiency advantage")
        
        # 5. FILE SIZE (Practical Factor)
        file_size = analysis.get("file_size_mb", 0)
        if file_size > 50:
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append("Large file size: Gemini handles big files better")
        
        # 6. COST OPTIMIZATION (Enhanced for multi-page)
        if total_pages > 15 or (total_pages > 8 and file_size > 20):
            original_model = strategy["recommended_model"]
            strategy["recommended_model"] = "gemini"
            strategy["reasoning"].append(f"Cost optimization: Switched from {original_model} to Gemini for large document")
        
        # 7. MULTI-PAGE PROCESSING STRATEGY
        if total_pages > 1:
            if strategy["recommended_model"] == "openai" and total_pages > 8:
                strategy["multi_page_strategy"] = "batch_with_page_fallback"
                strategy["reasoning"].append("OpenAI with page-by-page fallback for reliability")
            elif strategy["recommended_model"] == "gemini":
                strategy["multi_page_strategy"] = "batch_process"
                strategy["reasoning"].append("Gemini batch processing for efficiency")
        
        return strategy
    
    def _print_selection_reasoning(self, strategy: Dict, analysis: Dict, file_type: str):
        """Print detailed reasoning for model selection"""
        selected_model = strategy["recommended_model"]
        reasoning = strategy.get("reasoning", [])
        total_pages = analysis.get("total_pages", 1)
        
        print(f"🤖 Model Selection: {selected_model.upper()}")
        print(f"📄 Multi-page Strategy: {strategy.get('multi_page_strategy', 'standard')}")
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
            if total_pages > 5:
                print(f"   ⚠️  May be slower for {total_pages} pages")
            
        elif selected_model == "gemini":
            print(f"📈 Expected Strengths:")
            print(f"   • Faster processing speed")
            print(f"   • Better cost efficiency")
            print(f"   • Good performance on handwriting")
            print(f"   • Handles poor image quality well")
            if total_pages > 1:
                print(f"   • Efficient multi-page processing ({total_pages} pages)")
        
        # Multi-page specific expectations
        if total_pages > 1:
            print(f"📑 Multi-page Processing:")
            print(f"   • Total pages: {total_pages}")
            print(f"   • Strategy: {strategy.get('multi_page_strategy')}")
            print(f"   • Expected processing time: {'Medium' if total_pages <= 5 else 'Extended'}")