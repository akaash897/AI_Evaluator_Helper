import os
import asyncio
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import agentic components
from agents import ExamProcessingOrchestrator, DocumentAnalyzerAgent, QuestionExtractorAgent

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

# Initialize the agentic orchestrator
orchestrator = ExamProcessingOrchestrator()
print("Agentic orchestrator initialized")

async def extract_question_text(pdf_path: str) -> str:
    """Extract questions using pure agentic system"""
    print("🤖 Extracting questions with agentic system...")
    
    # Create analyzer and extractor agents
    analyzer = DocumentAnalyzerAgent()
    extractor = QuestionExtractorAgent()
    
    # Analyze document first
    analysis_task = {
        "file_path": pdf_path,
        "file_type": "question_paper"
    }
    analysis_result = await analyzer.execute(analysis_task)
    
    if not analysis_result.success:
        raise Exception(f"Document analysis failed: {analysis_result.error}")
    
    # Extract questions using the recommended strategy
    strategy = analysis_result.data["strategy"]
    print(f"🎯 Using model: {strategy['recommended_model']}")
    
    extraction_task = {
        "file_path": pdf_path,
        "strategy": strategy
    }
    extraction_result = await extractor.execute(extraction_task)
    
    if not extraction_result.success:
        raise Exception(f"Question extraction failed: {extraction_result.error}")
    
    print("✅ Question extraction successful!")
    return extraction_result.data["question_text"]



async def process_student_pdf(student_path: str, question_text: str, output_folder: str) -> str:
    """Process student PDF with pure agentic system"""
    print("🤖 Processing student PDF with agentic system...")
    
    # Use orchestrator for complete processing
    result = await orchestrator.process_exam_documents(
        question_pdf=None,  # Questions already extracted
        answer_pdf=student_path,
        output_folder=output_folder,
        selected_model="gemini"  # Will be overridden by analysis
    )
    
    if not result["success"]:
        raise Exception(f"Student processing failed: {result['error']}")
    
    print("✅ Student processing successful!")
    return result["pdf_filename"]



async def process_exam_documents_agentic(question_pdf: str, answer_pdf: str, output_folder: str, selected_model: str = "gemini"):
    """
    Direct agentic processing function for complete workflow
    """
    return await orchestrator.process_exam_documents(question_pdf, answer_pdf, output_folder, selected_model)