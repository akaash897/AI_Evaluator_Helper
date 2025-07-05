# app.py - Enhanced with agentic features
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session, jsonify
import json
import shutil
import asyncio
import traceback
from datetime import datetime
import os

# Import the main processing functions
from main import extract_question_text, process_student_pdf, process_exam_documents_agentic

UPLOAD_FOLDER = "uploads"
QUESTION_FOLDER = os.path.join(UPLOAD_FOLDER, "question_data")
STUDENT_FOLDER = os.path.join(UPLOAD_FOLDER, "students_data")
OUTPUT_FOLDER = "outputs"
TMP_FOLDER = "tmp"
TMP_CURRENT_FOLDER = os.path.join(TMP_FOLDER, "current")
FOLDERS_META_FILE = "folders_metadata.json"

# Create all required directories
os.makedirs(QUESTION_FOLDER, exist_ok=True)
os.makedirs(STUDENT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(TMP_CURRENT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

def load_folders_metadata():
    try:
        if os.path.exists(FOLDERS_META_FILE):
            with open(FOLDERS_META_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
    return {"question_folders": [], "student_folders": []}

def save_folders_metadata(metadata):
    try:
        with open(FOLDERS_META_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def scan_existing_folders():
    """Scan and update metadata with existing folders"""
    metadata = {"question_folders": [], "student_folders": []}
    
    if os.path.exists(QUESTION_FOLDER):
        for folder_name in os.listdir(QUESTION_FOLDER):
            folder_path = os.path.join(QUESTION_FOLDER, folder_name)
            if os.path.isdir(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                if files:
                    metadata["question_folders"].append({
                        "name": folder_name,
                        "files": files,
                        "created": datetime.now().isoformat()
                    })
    
    if os.path.exists(STUDENT_FOLDER):
        for folder_name in os.listdir(STUDENT_FOLDER):
            folder_path = os.path.join(STUDENT_FOLDER, folder_name)
            if os.path.isdir(folder_path):
                files = []
                for root, dirs, file_list in os.walk(folder_path):
                    for f in file_list:
                        if f.endswith('.pdf'):
                            rel_path = os.path.relpath(os.path.join(root, f), folder_path)
                            files.append(rel_path)
                if files:
                    metadata["student_folders"].append({
                        "name": folder_name,
                        "files": files,
                        "created": datetime.now().isoformat()
                    })
    
    save_folders_metadata(metadata)
    return metadata

def create_timestamped_folder(base_path, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}_{timestamp}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_name, folder_path

def copy_current_files_to_tmp(question_folder, question_file, student_folder, student_file):
    """Copy current processing files to tmp/current for display"""
    try:
        # Clear previous tmp files
        if os.path.exists(TMP_CURRENT_FOLDER):
            shutil.rmtree(TMP_CURRENT_FOLDER)
        os.makedirs(TMP_CURRENT_FOLDER, exist_ok=True)
        
        # Copy question file
        if question_folder and question_file:
            src_q = os.path.join(QUESTION_FOLDER, question_folder, question_file)
            if os.path.exists(src_q):
                dst_q = os.path.join(TMP_CURRENT_FOLDER, f"current_question.pdf")
                shutil.copy2(src_q, dst_q)
                print(f"Copied question file to tmp: {dst_q}")
        
        # Copy student file
        if student_folder and student_file:
            src_s = os.path.join(STUDENT_FOLDER, student_folder, student_file)
            if os.path.exists(src_s):
                dst_s = os.path.join(TMP_CURRENT_FOLDER, f"current_student.pdf")
                shutil.copy2(src_s, dst_s)
                print(f"Copied student file to tmp: {dst_s}")
                
    except Exception as e:
        print(f"Error copying files to tmp: {e}")

@app.route("/api/folders")
def get_folders():
    try:
        metadata = scan_existing_folders()
        return jsonify(metadata)
    except Exception as e:
        print(f"Error in get_folders: {e}")
        return jsonify({"question_folders": [], "student_folders": []})

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory("outputs", filename, as_attachment=True)

@app.route("/view/<filename>")
def view_pdf(filename):
    return send_from_directory("outputs", filename)

@app.route("/view_question/<folder>/<filename>")
def view_question_pdf(folder, filename):
    return send_from_directory(f"uploads/question_data/{folder}", filename)

@app.route("/view_student/<folder>/<filename>")
def view_student_pdf(folder, filename):
    return send_from_directory(f"uploads/students_data/{folder}", filename)

@app.route("/view_current_question")
def view_current_question():
    """View current question file from tmp"""
    return send_from_directory(TMP_CURRENT_FOLDER, "current_question.pdf")

@app.route("/view_current_student")
def view_current_student():
    """View current student file from tmp"""
    return send_from_directory(TMP_CURRENT_FOLDER, "current_student.pdf")

@app.route("/results")
def results():
    results = session.get('current_results', [])
    question_folder = session.get('question_folder', None)
    question_filename = session.get('question_filename', None)
    student_folder = session.get('student_folder', None)
    student_filename = session.get('student_filename', None)
    
    return render_template("results.html", 
                         results=results, 
                         question_folder=question_folder, 
                         question_filename=question_filename,
                         student_folder=student_folder,
                         student_filename=student_filename)

@app.route("/api/process_agentic", methods=["POST"])
def process_agentic_endpoint():
    """New endpoint for full agentic processing"""
    try:
        data = request.get_json()
        question_pdf = data.get("question_pdf")
        answer_pdf = data.get("answer_pdf")
        fallback_model = data.get("fallback_model", "gemini")
        
        # Run agentic processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            process_exam_documents_agentic(question_pdf, answer_pdf, OUTPUT_FOLDER, fallback_model)
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
            metadata = load_folders_metadata()
            generated_pdfs = []
            question_folder = None
            question_filename = None
            student_folder = None
            student_filename = None
            question_text = None
            
            # Get fallback model preference (automatic selection is primary)
            fallback_model = request.form.get('fallback_model', 'gemini')
            print(f"Using automatic model selection with {fallback_model} as fallback")
            
            # Handle question paper
            if request.form.get('question_option') == 'new':
                if "question_paper" in request.files:
                    q_file = request.files["question_paper"]
                    if q_file and q_file.filename:
                        print(f"Processing question file: {q_file.filename}")
                        folder_name, folder_path = create_timestamped_folder(QUESTION_FOLDER, "questions")
                        q_path = os.path.join(folder_path, q_file.filename)
                        q_file.save(q_path)
                        
                        print("Extracting question text with agentic system...")
                        question_text = extract_question_text(q_path, fallback_model)
                        question_folder = folder_name
                        question_filename = q_file.filename
                        
                        metadata["question_folders"].append({
                            "name": folder_name,
                            "files": [q_file.filename],
                            "created": datetime.now().isoformat()
                        })
                        print("Question processing complete")
            
            elif request.form.get('question_option') == 'existing':
                existing_folder = request.form.get('existing_question_folder')
                existing_file = request.form.get('existing_question_file')
                if existing_folder and existing_file:
                    print(f"Using existing question: {existing_folder}/{existing_file}")
                    q_path = os.path.join(QUESTION_FOLDER, existing_folder, existing_file)
                    if os.path.exists(q_path):
                        question_text = extract_question_text(q_path, fallback_model)
                        question_folder = existing_folder
                        question_filename = existing_file
                        print("Existing question loaded")
                    else:
                        print(f"Question file not found: {q_path}")

            if not question_text:
                print("ERROR: No question text extracted")
                return "Error: Could not process question paper", 400

            # Handle student PDFs
            if request.form.get('student_option') == 'new':
                if "student_pdfs" in request.files:
                    student_files = request.files.getlist("student_pdfs")
                    selected_pdf = request.form.get('selected_student_pdf')
                    
                    if student_files and selected_pdf:
                        print(f"Processing {len(student_files)} student files, selected: {selected_pdf}")
                        folder_name, folder_path = create_timestamped_folder(STUDENT_FOLDER, "students")
                        file_list = []
                        
                        for s_file in student_files:
                            if s_file and s_file.filename:
                                clean_filename = os.path.basename(s_file.filename)
                                s_path = os.path.join(folder_path, clean_filename)
                                s_file.save(s_path)
                                file_list.append(clean_filename)
                                print(f"Saved: {clean_filename}")
                                
                                if clean_filename == selected_pdf:
                                    print(f"Processing selected PDF: {clean_filename}")
                                    main_path = os.path.join(STUDENT_FOLDER, clean_filename)
                                    shutil.copy2(s_path, main_path)
                                    
                                    pdf_filename = process_student_pdf(clean_filename, question_text, OUTPUT_FOLDER, fallback_model)
                                    if pdf_filename:
                                        generated_pdfs.append(pdf_filename)
                                        student_folder = folder_name
                                        student_filename = clean_filename
                                        print(f"Generated: {pdf_filename}")
                                    else:
                                        print(f"Failed to generate PDF for: {clean_filename}")
                                    
                                    try:
                                        os.remove(main_path)
                                    except:
                                        pass
                        
                        metadata["student_folders"].append({
                            "name": folder_name,
                            "files": file_list,
                            "created": datetime.now().isoformat()
                        })
            
            elif request.form.get('student_option') == 'existing':
                existing_folder = request.form.get('existing_student_folder')
                selected_pdf = request.form.get('selected_existing_student_pdf')
                if existing_folder and selected_pdf:
                    print(f"Using existing student: {existing_folder}/{selected_pdf}")
                    s_path = os.path.join(STUDENT_FOLDER, existing_folder, selected_pdf)
                    if os.path.exists(s_path):
                        temp_file = os.path.basename(selected_pdf)
                        temp_path = os.path.join(STUDENT_FOLDER, temp_file)
                        shutil.copy2(s_path, temp_path)
                        
                        pdf_filename = process_student_pdf(temp_file, question_text, OUTPUT_FOLDER, fallback_model)
                        if pdf_filename:
                            generated_pdfs.append(pdf_filename)
                            student_folder = existing_folder
                            student_filename = selected_pdf
                            print(f"Generated: {pdf_filename}")
                        else:
                            print(f"Failed to generate PDF for: {temp_file}")
                        
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    else:
                        print(f"Student file not found: {s_path}")

            save_folders_metadata(metadata)
            
            # Copy current files to tmp for display
            copy_current_files_to_tmp(question_folder, question_filename, student_folder, student_filename)
            
            # Cleanup outputs folder
            try:
                for file in os.listdir(OUTPUT_FOLDER):
                    if file.endswith(('.aux', '.log', '.tex', '.fdb_latexmk', '.fls', '.synctex.gz')):
                        try:
                            os.remove(os.path.join(OUTPUT_FOLDER, file))
                        except:
                            pass
            except Exception as e:
                print(f"Error during cleanup: {e}")

            session['current_results'] = generated_pdfs
            session['question_folder'] = question_folder
            session['question_filename'] = question_filename
            session['student_folder'] = student_folder
            session['student_filename'] = student_filename
            
            print(f"Processing complete. Generated {len(generated_pdfs)} PDFs")
            return redirect(url_for('results'))
            
        except Exception as e:
            print(f"ERROR in processing: {e}")
            print(traceback.format_exc())
            return f"Error processing files: {str(e)}", 500

    return render_template("index.html")

if __name__ == "__main__":
    print("Starting Agentic Exam Processing System...")
    print("System features:")
    print("- Automatic AI model selection based on document analysis")
    print("- Self-healing error recovery")
    print("- Detailed workflow monitoring")
    print("- Fallback to original methods if agentic fails")
    app.run(debug=True)