import os
import shutil
import zipfile
from flask import Flask, request, send_file, jsonify, send_from_directory, render_template, session
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import json
from flask_cors import CORS
import datetime
from io import BytesIO
import time
import threading
import traceback  # Import for better error handling

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
app.secret_key = 'xr_lab_autograder_secret_key'  # For session management

# Enable CORS for all routes and all origins
CORS(app)

app.static_folder = 'static'

UPLOAD_FOLDER = 'uploads'
REDUCED_FOLDER = 'results/reduced_pdfs'
RESULTS_FOLDER = 'results'
EXTRACTED_IMAGES_FOLDER = 'Extracted_images'
META_FILE_CSV = os.path.join(RESULTS_FOLDER, 'image_metadata.csv')
META_SIZE_CSV = os.path.join(RESULTS_FOLDER, 'file_size_metadata.csv')

# Global progress tracker
processing_status = {
    "is_processing": False,
    "progress": 0,
    "status_message": "",
    "start_time": None,
    "end_time": None,
    "processed_files": [],
    "total_files": 0,
    "current_file": "",
    "last_update": None,
    "error": None  # Add error field to track any errors
}

# Create necessary folders
for folder in [UPLOAD_FOLDER, REDUCED_FOLDER, RESULTS_FOLDER, 'static', EXTRACTED_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def update_progress(percentage, message="", current_file=""):
    """Update processing progress"""
    global processing_status
    processing_status["progress"] = int(percentage)  # Ensure integer value
    if message:
        processing_status["status_message"] = message
    if current_file:
        processing_status["current_file"] = current_file
    processing_status["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Progress: {percentage}% - {message}")

def reset_processing_status():
    """Reset the processing status when processing is complete or fails"""
    global processing_status
    processing_status["is_processing"] = False
    processing_status["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_extracted_images():
    """Clean up the extracted images folder"""
    if os.path.exists(EXTRACTED_IMAGES_FOLDER):
        shutil.rmtree(EXTRACTED_IMAGES_FOLDER)
        os.makedirs(EXTRACTED_IMAGES_FOLDER, exist_ok=True)
        print(f"Cleaned up {EXTRACTED_IMAGES_FOLDER} folder")

def extract_images_from_pdf(pdf_path, output_dir, min_width=1200):
    """
    Extract images from a PDF file and save them to the output directory.
    Returns a list of records with metadata for each extracted image.
    """
    records = []
    
    try:
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        doc = fitz.open(pdf_path)
        
        # Create subfolder for this PDF's images
        pdf_img_dir = os.path.join(output_dir, base_name)
        os.makedirs(pdf_img_dir, exist_ok=True)
        
        print(f"Extracting images from {filename} ({len(doc)} pages)...")
        update_progress(processing_status["progress"], f"Extracting images from PDF", filename)
        
        for page_idx in range(len(doc)):
            # Only update the status message, not the percentage
            current_progress = processing_status["progress"]
            update_progress(current_progress, f"Processing page {page_idx+1}/{len(doc)}", filename)
            
            page = doc[page_idx]
            images = page.get_images(full=True)
            
            if not images:
                continue
                
            for img_idx, img_info in enumerate(images, start=1):
                xref = img_info[0]
                img_data = doc.extract_image(xref)
                img_bytes = img_data["image"]
                
                try:
                    img = Image.open(BytesIO(img_bytes))
                except Exception as e:
                    print(f"Couldn't open image on page {page_idx+1}, idx {img_idx}: {e}")
                    continue
                    
                w, h = img.size
                img_name = f"{base_name}_p{page_idx+1}_img{img_idx}.png"
                img_path = os.path.join(pdf_img_dir, img_name)
                img.save(img_path)
                
                records.append({
                    "pdf_file": filename,
                    "page": page_idx + 1,
                    "image_name": img_name,
                    "image_path": os.path.join(EXTRACTED_IMAGES_FOLDER, base_name, img_name),
                    "width": w,
                    "height": h,
                    "resolution": f"{w}x{h}",
                    "is_high_res": w >= min_width
                })
                
        doc.close()
        return records
        
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")
        return []

def compress_pdf(input_path, output_path, dpi=100, downscale_factor=2):
    try:
        doc = fitz.open(input_path)
        new_pdf = fitz.open()
        
        filename = os.path.basename(input_path)
        update_progress(processing_status["progress"], f"Compressing PDF file", filename)
        
        for page_idx, page in enumerate(doc):
            # Only update the status message, not the percentage
            current_progress = processing_status["progress"]
            update_progress(current_progress, f"Compressing page {page_idx+1}/{len(doc)}", filename)
            
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            new_size = (pix.width // downscale_factor, pix.height // downscale_factor)
            img = img.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PDF", resolution=dpi)
            buffer.seek(0)
            img_pdf = fitz.open("pdf", buffer)
            new_pdf.insert_pdf(img_pdf)
        new_pdf.save(output_path)
        new_pdf.close()
        doc.close()
        return True
    except Exception as e:
        print(f"Failed to compress {input_path}: {e}")
        return False

def process_files_background(saved_files):
    """Process files in background"""
    global processing_status
    
    try:
        processing_status["is_processing"] = True
        processing_status["start_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        processing_status["progress"] = 0
        processing_status["status_message"] = "Starting file processing..."
        processing_status["total_files"] = len(saved_files)
        processing_status["processed_files"] = []
        processing_status["error"] = None
        
        # Clean up extracted images folder
        clean_extracted_images()
        
        # Update progress to 2% after cleanup
        update_progress(2, "Cleaned upload folder, preparing to process files...")
        
        # Extract images from PDFs
        all_image_records = []
        
        # Calculate total pages to track progress
        total_pages = 0
        pdf_count = 0
        for file_info in saved_files:
            if file_info['path'].lower().endswith('.pdf'):
                pdf_count += 1
                try:
                    doc = fitz.open(file_info['path'])
                    total_pages += len(doc)
                    doc.close()
                except:
                    pass
        
        # If no PDFs or no pages, avoid division by zero
        if total_pages == 0:
            total_pages = 1
            
        processed_pages = 0
        
        for file_idx, file_info in enumerate(saved_files):
            file_path = file_info['path']
            if file_path.lower().endswith('.pdf'):
                print(f"Extracting images from {file_info['name']}")
                
                # Update progress at the start of each file
                file_progress = 5 + int((file_idx / len(saved_files)) * 5)
                update_progress(file_progress, f"Preparing to extract images {file_idx+1}/{len(saved_files)}", file_info['name'])
                
                # Extract images
                doc = fitz.open(file_path)
                num_pages = len(doc)
                doc.close()
                
                records = extract_images_from_pdf(file_path, EXTRACTED_IMAGES_FOLDER)
                all_image_records.extend(records)
                processing_status["processed_files"].append(file_info['name'])
                
                # Update processed pages count
                processed_pages += num_pages
                
                # Calculate extraction progress (5-55%)
                extraction_progress = 10 + int((processed_pages / total_pages) * 45)
                update_progress(extraction_progress, f"Completed image extraction {processed_pages}/{total_pages} pages", file_info['name'])
        
        # Save image metadata
        if all_image_records:
            update_progress(55, f"Image extraction complete, {len(all_image_records)} images total")
            df_images = pd.DataFrame(all_image_records)
            os.makedirs(os.path.dirname(META_FILE_CSV), exist_ok=True)
            df_images.to_csv(META_FILE_CSV, index=False)
            print(f"Saved metadata for {len(all_image_records)} images to {META_FILE_CSV}")
        else:
            update_progress(55, "No images were extracted from PDFs")
            print("No images were extracted from PDFs")
        
        update_progress(60, "Starting PDF compression...")
        
        # Process and compress PDFs
        size_results = []
        reduced_pdfs = []
        total_original_size = 0
        total_reduced_size = 0
        
        for file_idx, file_info in enumerate(saved_files):
            filename = file_info['name']
            file_path = file_info['path']
            original_size = file_info['size']
            total_original_size += original_size
            
            # Calculate compression progress (60-95%)
            compress_progress = 60 + int((file_idx / len(saved_files)) * 35)
            update_progress(compress_progress, f"Compressing {file_idx+1}/{len(saved_files)}", filename)
            
            if filename.lower().endswith('.pdf'):
                reduced_path = os.path.join(REDUCED_FOLDER, 'reduced_' + filename)
                # Ensure reduced folder exists
                os.makedirs(os.path.dirname(reduced_path), exist_ok=True)
                success = compress_pdf(file_path, reduced_path)
                
                if success and os.path.exists(reduced_path):
                    reduced_size = os.path.getsize(reduced_path)
                    total_reduced_size += reduced_size
                    status = 'Reduced'
                    
                    # Get original and compressed dimensions
                    try:
                        orig_doc = fitz.open(file_path)
                        orig_page = orig_doc[0]
                        orig_width, orig_height = orig_page.rect.width, orig_page.rect.height
                        orig_doc.close()
                        
                        new_doc = fitz.open(reduced_path)
                        new_page = new_doc[0]
                        new_width, new_height = new_page.rect.width, new_page.rect.height
                        new_doc.close()
                    except:
                        orig_width, orig_height = 0, 0
                        new_width, new_height = 0, 0
                    
                    reduction_percent = round(100 * (1 - reduced_size / original_size), 1)
                    
                    # Add to results
                    size_results.append({
                        'file_name': filename,
                        'original_path': file_path,
                        'reduced_path': reduced_path,
                        'original_size_kb': round(original_size / 1024, 1),
                        'reduced_size_kb': round(reduced_size / 1024, 1),
                        'reduction_percent': reduction_percent,
                        'original_resolution': f"{int(orig_width)}x{int(orig_height)}",
                        'reduced_resolution': f"{int(new_width)}x{int(new_height)}",
                        'status': status,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    reduced_pdfs.append({
                        'original_name': filename,
                        'reduced_name': 'reduced_' + filename,
                        'download_url': f'/download/reduced/reduced_{filename}'
                    })
                else:
                    reduced_size = original_size
                    total_reduced_size += reduced_size
                    status = 'Failed'
                    
                    size_results.append({
                        'file_name': filename,
                        'original_path': file_path,
                        'reduced_path': '-',
                        'original_size_kb': round(original_size / 1024, 1),
                        'reduced_size_kb': round(original_size / 1024, 1),
                        'reduction_percent': 0,
                        'original_resolution': 'Unknown',
                        'reduced_resolution': 'Unknown',
                        'status': status,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            else:
                # List non-PDF files as-is
                size_results.append({
                    'file_name': filename,
                    'original_path': file_path,
                    'reduced_path': '-',
                    'original_size_kb': round(original_size / 1024, 1),
                    'reduced_size_kb': round(original_size / 1024, 1),
                    'reduction_percent': 0,
                    'original_resolution': 'N/A',
                    'reduced_resolution': 'N/A',
                    'status': 'Not Processed (Not a PDF)',
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                total_reduced_size += original_size
        
        update_progress(95, "Generating result data...")
        
        # Save size reduction metadata
        if size_results:
            df_sizes = pd.DataFrame(size_results)
            os.makedirs(os.path.dirname(META_SIZE_CSV), exist_ok=True)
            df_sizes.to_csv(META_SIZE_CSV, index=False)
            
            # Create zip with all reduced PDFs
            zip_path = os.path.join(RESULTS_FOLDER, 'reduced_pdfs.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in os.listdir(REDUCED_FOLDER):
                    file_path = os.path.join(REDUCED_FOLDER, f)
                    if os.path.isfile(file_path):
                        zf.write(file_path, f)
            
            # Calculate statistics
            reduced_count = sum(1 for r in size_results if r['status'] == 'Reduced')
            failed_count = sum(1 for r in size_results if r['status'] == 'Failed')
            not_processed = sum(1 for r in size_results if r['status'] == 'Not Processed (Not a PDF)')
            
            percent_saved = 0
            if total_original_size > 0:
                percent_saved = round(100 * (total_original_size - total_reduced_size) / total_original_size)
                
            # Save results for later retrieval
            summary = {
                'total_files': len(size_results),
                'reduced_count': reduced_count,
                'failed_count': failed_count,
                'not_processed_count': not_processed,
                'total_original_size_kb': round(total_original_size / 1024, 1),
                'total_reduced_size_kb': round(total_reduced_size / 1024, 1),
                'percent_saved': percent_saved,
                'image_extraction': {
                    'total_images': len(all_image_records),
                    'high_res_images': sum(1 for img in all_image_records if img['is_high_res']),
                    'metadata_csv': META_FILE_CSV
                },
                'size_results': size_results,
                'image_metadata': all_image_records[:10],  # First 10 as preview
                'meta_size_csv_path': META_SIZE_CSV,
                'meta_image_csv_path': META_FILE_CSV,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(RESULTS_FOLDER, 'latest_results.json'), 'w') as f:
                json.dump(summary, f)
        
        update_progress(100, "Processing complete!")
        reset_processing_status()  # Reset processing status
    
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print("ERROR:", error_msg)
        print(tb)
        
        processing_status["status_message"] = f"Processing error: {error_msg}"
        processing_status["error"] = error_msg
        reset_processing_status()  # Reset processing status on error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/progress')
def get_progress():
    """Get current processing progress"""
    return jsonify(processing_status)

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        print("Received upload request")
        
        # Check if files exist
        if 'files' not in request.files:
            print("No files part")
            return jsonify({'status': 'error', 'message': 'No files part'}), 400
            
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            print("No files selected")
            return jsonify({'status': 'error', 'message': 'No files selected'}), 400
        
        print(f"Processing {len(files)} files")
        
        # Save files first in the main thread
        saved_file_paths = []
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            saved_file_paths.append({
                'name': filename,
                'path': file_path,
                'size': os.path.getsize(file_path)
            })
        
        # Start background processing thread
        if not processing_status["is_processing"]:
            threading.Thread(target=process_files_background, args=(saved_file_paths,)).start()
            return jsonify({'status': 'success', 'message': 'Files uploaded successfully, processing started in background'})
        else:
            return jsonify({'status': 'error', 'message': 'A processing task is already in progress'}), 400
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print("ERROR:", error_msg)
        print(tb)
        return jsonify({'status': 'error', 'message': error_msg}), 500

@app.route('/check_metadata_status')
def check_metadata_status():
    """Check if the metadata file exists and return its information"""
    try:
        response = {
            'exists': False,
            'status': processing_status.get('status_message', ''),
            'is_processing': processing_status.get('is_processing', False),
            'progress': processing_status.get('progress', 0),
            'current_file': processing_status.get('current_file', '')
        }
        
        if os.path.exists(META_FILE_CSV):
            file_stats = os.stat(META_FILE_CSV)
            file_size = file_stats.st_size
            mod_time = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # If file exists, try to read the number of rows
            try:
                df = pd.read_csv(META_FILE_CSV)
                row_count = len(df)
                high_res_count = df['is_high_res'].sum() if 'is_high_res' in df.columns else 0
            except:
                row_count = 0
                high_res_count = 0
                
            response.update({
                'exists': True,
                'file_size': file_size,
                'file_size_kb': round(file_size / 1024, 1),
                'last_modified': mod_time,
                'row_count': row_count,
                'high_res_count': int(high_res_count)
            })
            
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'exists': False, 
            'error': str(e),
            'status': 'Error checking metadata: ' + str(e),
            'is_processing': False
        })

@app.route('/download/image_metadata.csv')
def download_image_csv():
    return send_file(META_FILE_CSV, as_attachment=True, download_name='image_metadata.csv')

@app.route('/download/file_size_metadata.csv')
def download_size_csv():
    return send_file(META_SIZE_CSV, as_attachment=True, download_name='file_size_metadata.csv')

@app.route('/download/reduced_pdfs.zip')
def download_zip():
    zip_path = os.path.join(RESULTS_FOLDER, 'reduced_pdfs.zip')
    return send_file(zip_path, as_attachment=True)

@app.route('/download/reduced/<filename>')
def download_reduced_pdf(filename):
    return send_from_directory(REDUCED_FOLDER, filename, as_attachment=True)

@app.route('/latest_results')
def latest_results():
    try:
        results_path = os.path.join(RESULTS_FOLDER, 'latest_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            # Add a status field for the client
            data['status'] = 'success'
            
            # Add paths for client-side use
            data['pdf_path'] = '/download/reduced_pdfs.zip'
            data['csv_path'] = '/download/image_metadata.csv'
            
            # Add size info if available
            if 'size_results' in data and len(data['size_results']) > 0:
                total_original_size = sum(item['original_size_kb'] for item in data['size_results']) * 1024
                total_reduced_size = sum(item['reduced_size_kb'] for item in data['size_results']) * 1024
                data['original_size'] = total_original_size
                data['compressed_size'] = total_reduced_size
                
            # Add image count if available
            if 'image_extraction' in data:
                data['image_count'] = data['image_extraction'].get('total_images', 0)
                
            return jsonify(data)
        else:
            return jsonify({
                'status': 'error',
                'message': 'No results available yet'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving results: {str(e)}'
        })

@app.route('/test_upload', methods=['POST'])
def test_upload():
    try:
        # Check if the request has files
        if 'files' not in request.files:
            return jsonify({'status': 'error', 'message': 'No files part in the request'}), 400
            
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'status': 'error', 'message': 'No files selected'}), 400
            
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({'status': 'error', 'message': f'File {file.filename} is not a PDF'}), 400
        
        # All validation passed
        return jsonify({'status': 'success', 'message': 'Files validated successfully'}), 200
    except Exception as e:
        error_msg = str(e)
        print("Test upload error:", error_msg)
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001, host="0.0.0.0")
