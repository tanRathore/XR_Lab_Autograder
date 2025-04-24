from flask import Flask, request, jsonify, send_from_directory, Response
import traceback 
from flask_cors import CORS  # Import CORS
import fitz
from PIL import Image
from io import BytesIO
import os
import json
from datetime import datetime
import hashlib
from functools import wraps
from werkzeug.utils import secure_filename
from autograder_logic import run_autograder_full, text_model, vision_model, extract_text_from_pdf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = "/tmp/autograder_uploads"
SUBMISSIONS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submissions')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBMISSIONS_FOLDER, exist_ok=True)

# Admin credentials (in production, use environment variables)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"  # Change this in production!

# Function to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated_function

# Function to check authentication
def check_auth(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# Function to send authentication request
def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

# Function to save submission data
def save_submission(student_name, student_pid, architect_name, grade, score, rubric_scores, detailed_evaluation):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_pid}_{timestamp}.json"
    filepath = os.path.join(SUBMISSIONS_FOLDER, filename)
    
    submission_data = {
        "student_name": student_name,
        "student_pid": student_pid,
        "architect_name": architect_name,
        "timestamp": timestamp,
        "grade": grade,
        "score": score,
        "rubric_scores": rubric_scores,
        "detailed_evaluation": detailed_evaluation
    }
    
    with open(filepath, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    return filepath

# Function to get all submissions
def get_all_submissions():
    submissions = []
    for filename in os.listdir(SUBMISSIONS_FOLDER):
        if filename.endswith('.json'):
            filepath = os.path.join(SUBMISSIONS_FOLDER, filename)
            with open(filepath, 'r') as f:
                submission = json.load(f)
                submissions.append(submission)
    
    # Sort by timestamp (newest first)
    submissions.sort(key=lambda x: x['timestamp'], reverse=True)
    return submissions

@app.route("/", methods=["GET"])
def homepage():
    return "<h2> Welcome to the XR Autograder</h2><p>Please submit your assignment through the frontend.</p>"

@app.route("/a", methods=["GET"])
def serve_frontend():
    return send_from_directory(".", "frontend.html")

@app.route("/admin", methods=["GET"])
@login_required
def admin_page():
    return send_from_directory(".", "admin.html")

@app.route("/api/submissions", methods=["GET"])
@login_required
def get_submissions():
    submissions = get_all_submissions()
    return jsonify(submissions)

@app.route("/", methods=["POST"])
def grade_student():
    student_name = request.form.get("name")
    student_pid = request.form.get("pid")
    architect_name = request.form.get("architect", "Bjarke Ingels")
    uploaded_file = request.files.get("file")

    if not uploaded_file or not uploaded_file.filename.endswith(".pdf"):
        return jsonify({"error": "No PDF file uploaded."}), 400

    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(filepath)

    try:
        # Extract text from PDF
        text = extract_text_from_pdf(filepath)
        
        # Run the autograder to get the scores
        result = run_autograder_full(filepath, architect_name=architect_name, debug=False)
        
        # Get the detailed evaluation text from the result
        detailed_evaluation_text = result.get("detailed_evaluation", "No detailed evaluation available.")

        feedback_prompt = f"""
You are writing feedback for {student_name} (PID: {student_pid}) on their architecture submission.
Summarize their performance across categories, highlight strengths, and offer suggestions for improvement.
Their final score is {result['final_percent']}% and grade is {result['grade']}.
"""

        gemini_feedback = text_model.generate_content([feedback_prompt]).text

        # Save submission data
        save_submission(
            student_name=student_name,
            student_pid=student_pid,
            architect_name=architect_name,
            grade=result['grade'],
            score=result['final_percent'],
            rubric_scores=result['rubric_scores'],
            detailed_evaluation=result['detailed_evaluation']
        )

        os.remove(filepath)  

        return jsonify({
            "feedback": gemini_feedback,
            "detailed_evaluation": detailed_evaluation_text,
            "score": result["final_percent"],
            "grade": result["grade"],
            "rubric_scores": result["rubric_scores"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/grade', methods=['POST'])
def grade_submission():
    try:
        data = request.get_json()
        student_name = data.get('student_name')
        student_pid = data.get('student_pid')
        architect_name = data.get('architect_name')
        pdf_path = data.get('pdf_path')
        
        if not all([student_name, student_pid, architect_name, pdf_path]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Run the autograder
        result = run_autograder_full(pdf_path, architect_name)
        
        # Save the submission
        save_submission(
            student_name=student_name,
            student_pid=student_pid,
            architect_name=architect_name,
            grade=result['grade'],
            score=result['final_percent'],
            rubric_scores=result['rubric_scores'],
            detailed_evaluation=result['detailed_evaluation']
        )
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in grade_submission: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
