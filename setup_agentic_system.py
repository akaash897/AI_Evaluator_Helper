# setup_agentic_system.py
"""
Run this script to automatically set up your agentic exam processing system
"""
import os

def create_directory_structure():
    """Create all required directories"""
    directories = [
        "agents",
        "utils", 
        "templates",
        "uploads/question_data",
        "uploads/students_data",
        "outputs",
        "tmp/current"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    
    # agents/__init__.py
    agents_init = '''from .base_agent import BaseAgent, AgentResult
from .document_analyzer import DocumentAnalyzerAgent
from .question_extractor import QuestionExtractorAgent
from .answer_processor import AnswerProcessorAgent
from .latex_compiler import LatexCompilerAgent
from .orchestrator import ExamProcessingOrchestrator

__all__ = [
    'BaseAgent',
    'AgentResult', 
    'DocumentAnalyzerAgent',
    'QuestionExtractorAgent',
    'AnswerProcessorAgent',
    'LatexCompilerAgent',
    'ExamProcessingOrchestrator'
]'''
    
    with open("agents/__init__.py", "w") as f:
        f.write(agents_init)
    print("✅ Created agents/__init__.py")
    
    # utils/__init__.py (empty)
    with open("utils/__init__.py", "w") as f:
        f.write("# Empty file to make utils a package\n")
    print("✅ Created utils/__init__.py")

def create_sample_env():
    """Create a sample .env file if it doesn't exist"""
    if not os.path.exists(".env"):
        env_content = '''# Add your API keys here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
'''
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created sample .env file - ADD YOUR API KEYS!")
    else:
        print("✅ .env file already exists")

def update_requirements():
    """Update requirements.txt with all needed dependencies"""
    requirements = '''protobuf~=6.31.1
Flask~=3.1.1
Werkzeug~=3.1.3
dotenv~=0.9.9
python-dotenv~=1.1.0
langchain~=0.3.25
google-cloud-vision
google-cloud-storage
google-generativeai
openai
pillow
pdf2image
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Updated requirements.txt")

def check_existing_files():
    """Check which files from old setup need to be copied"""
    required_files = [
        "utils/ocr_openai.py",
        "utils/ocr_gemini.py", 
        "templates/index.html",
        "templates/results.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print("\n⚠️  MISSING FILES - Please copy from your old setup:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    return missing_files

def show_agent_import_fixes():
    """Show how to fix import errors in agent files"""
    print("\n🔧 TO FIX IMPORT ERRORS IN YOUR AGENT FILES:")
    print("Add these imports to the top of each agent file:\n")
    
    fixes = {
        "agents/document_analyzer.py": [
            "import os",
            "from typing import Dict, List, Any", 
            "from pdf2image import convert_from_path",
            "from PIL import Image",
            "from .base_agent import BaseAgent, AgentResult"
        ],
        "agents/question_extractor.py": [
            "from typing import Dict, List, Any",
            "from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex",
            "from utils.ocr_gemini import gemini_extract_question_text",
            "from .base_agent import BaseAgent, AgentResult"
        ],
        "agents/answer_processor.py": [
            "from typing import Dict, List, Any",
            "from utils.ocr_openai import pdf_to_images, gpt4o_extract_answer_latex", 
            "from utils.ocr_gemini import gemini_extract_answer_latex",
            "from .base_agent import BaseAgent, AgentResult"
        ],
        "agents/latex_compiler.py": [
            "import subprocess",
            "import os", 
            "import re",
            "from typing import Dict, Any",
            "from .base_agent import BaseAgent, AgentResult"
        ],
        "agents/orchestrator.py": [
            "import os",
            "import shutil",
            "from typing import Dict, Any",
            "from datetime import datetime",
            "from .base_agent import BaseAgent, AgentResult",
            "from .document_analyzer import DocumentAnalyzerAgent",
            "from .question_extractor import QuestionExtractorAgent", 
            "from .answer_processor import AnswerProcessorAgent",
            "from .latex_compiler import LatexCompilerAgent"
        ]
    }
    
    for file_path, imports in fixes.items():
        print(f"\n📁 {file_path}:")
        print("Add these imports at the top:")
        for imp in imports:
            print(f"   {imp}")

def main():
    print("🚀 Setting up Agentic Exam Processing System...\n")
    
    # Step 1: Create directories
    print("📁 Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Create __init__.py files
    print("\n📦 Creating Python package files...")
    create_init_files()
    
    # Step 3: Create .env file
    print("\n🔑 Setting up environment...")
    create_sample_env()
    
    # Step 4: Update requirements
    print("\n📋 Updating requirements...")
    update_requirements()
    
    # Step 5: Check for missing files
    print("\n🔍 Checking for existing files...")
    missing_files = check_existing_files()
    
    # Step 6: Show import fixes
    show_agent_import_fixes()
    
    # Final instructions
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📝 NEXT STEPS:")
    print("1. Copy missing files from your old setup (see list above)")
    print("2. Add the import statements to your agent files (see fixes above)")
    print("3. Add your API keys to the .env file")
    print("4. Install dependencies: pip install -r requirements.txt")
    print("5. Run the system: python app.py")
    
    print("\n✨ FEATURES YOU'LL GET:")
    print("• Intelligent model selection based on document quality")
    print("• Automatic retry with fallback strategies")
    print("• Self-healing error recovery") 
    print("• Detailed workflow monitoring")
    print("• 100% backward compatibility with your existing system")
    
    print("\n🔧 The system will automatically:")
    print("• Analyze document quality and choose the best AI model")
    print("• Retry failed operations with different approaches")
    print("• Fall back to your original methods if agentic processing fails")
    print("• Log detailed information about each processing step")

if __name__ == "__main__":
    main()