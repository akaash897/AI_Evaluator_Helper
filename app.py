# app.py - Pure agentic workflow
from flask import Flask, request, render_template, send_from_directory, jsonify
import asyncio
import os
import shutil

# Import the main processing functions
from main import extract_question_text, process_exam_documents_agentic

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

def cleanup_temp_folders():
    """Clean up temporary folders before processing new files"""
    temp_folders = ["tmp", "uploads"]
    for folder in temp_folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"🧹 Cleaned up {folder} directory")
            except Exception as e:
                print(f"⚠️ Could not clean {folder}: {e}")
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created fresh {folder} directory")


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route("/view/<filename>")
def view_pdf(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/api/process_agentic", methods=["POST"])
def process_agentic_endpoint():
    """API endpoint for agentic processing"""
    try:
        data = request.get_json()
        question_pdf = data.get("question_pdf")
        answer_pdf = data.get("answer_pdf")
        selected_model = data.get("selected_model", "gemini")
        
        result = asyncio.run(
            process_exam_documents_agentic(question_pdf, answer_pdf, OUTPUT_FOLDER, selected_model)
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get uploaded files
            question_file = request.files.get("question_paper")
            student_file = request.files.get("student_pdf")
            
            if not question_file or not student_file:
                return "Both question paper and student PDF are required", 400
            
            # Clean up old files first
            cleanup_temp_folders()
            
            # Save files temporarily
            q_path = os.path.join(UPLOAD_FOLDER, "question.pdf")
            s_path = os.path.join(UPLOAD_FOLDER, "student.pdf")
            
            question_file.save(q_path)
            student_file.save(s_path)
            
            print(f"📄 Processing: {question_file.filename} & {student_file.filename}")
            
            # Process with agentic system (let analyzer choose best model)
            result = asyncio.run(
                process_exam_documents_agentic(q_path, s_path, OUTPUT_FOLDER, "auto")
            )
            
            if result["success"]:
                print(f"✅ Processing successful: {result['pdf_filename']}")
                return render_template("results.html", 
                                     results=[result["pdf_filename"]],
                                     success=True)
            else:
                print(f"❌ Processing failed: {result['error']}")
                return f"Processing error: {result['error']}", 500
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return f"Error processing files: {str(e)}", 500

    return render_template("index.html")

if __name__ == "__main__":
    print("Starting Pure Agentic Exam Processing System...")
    print("Features:")
    print("- Automatic AI model selection")
    print("- Complete agentic workflow")
    print("- Self-healing error recovery")
    app.run(debug=True)