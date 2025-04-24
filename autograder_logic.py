__all__ = [
    "run_autograder_full",
    "extract_text_from_pdf",
    "extract_images_from_pdf",
    "evaluate_images_with_gemini",
    "evaluate_image_structure_and_captions",
    "gemini_detailed_rubric_eval",
    "generate_detailed_scorecard"
]
import os
import re
import json
import fitz  
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import spacy
import google.generativeai as genai
from dotenv import load_dotenv
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel("gemini-2.0-flash")
vision_model = genai.GenerativeModel("gemini-2.0-flash")
rubric = {
    "architect_chosen": 5,
    "doc_and_slides": 5,
    "bio_750_words": 5,
    "bio_structure": 5,
    "bio_references": 5,
    "10_buildings_with_images": 5,
    "image_quality": 5,
    "image_citations": 5,
    "image_relevance": 5,
    "personal_bio_photo": 5,
    "presentation_polish": 5
}

rubric_descriptions = {
    "architect_chosen": "Is the architect selected from Book Two and clearly identified?",
    "doc_and_slides": "Is the document structured well with table of contents and all required sections?",
    "bio_750_words": "Does the biography meet the 750-word requirement?",
    "bio_structure": "Does the biography cover who they are, where they studied, etc.?",
    "bio_references": "Are there 5–10 APA references with DOIs and citation counts?",
    "10_buildings_with_images": "Are 10 buildings covered with names, locations, significance, and image suggestions?",
    "image_quality": "Are the images high-resolution and well-composed?",
    "image_citations": "Do all images have proper attribution (photographer/source)?",
    "image_relevance": "Do images clearly relate to the architect's work?",
    "personal_bio_photo": "Is a professional student photo and 1–2 sentence bio included?",
    "presentation_polish": "Is the document polished, well-formatted, and web-publishable?"
}
# pdf_path = "/Users/tanishqsingh/Desktop/XR_Lab/cogs160submisson1.pdf"
def extract_text_from_pdf(pdf_path):
    print(f" Extracting text from: {pdf_path}")
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    print(" Extracted text from PDF")
    return text
def extract_images_from_pdf(pdf_path, min_width=1200, save_folder="/Users/tanishqsingh/Desktop/XR_Lab/Extracted_images"):
    print(f" Extracting images from: {pdf_path}")
    doc = fitz.open(pdf_path)
    os.makedirs(save_folder, exist_ok=True)
    image_data = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(BytesIO(image_bytes))
            width, height = img_pil.size
            img_pil.save(os.path.join(save_folder, f"page{page_index+1}_img{img_index+1}.png"))
            image_data.append({
                "page": page_index + 1,
                "width": width,
                "height": height,
                "coordinates": img[1:5],
                "image": img_pil,
                "filename": f"page{page_index+1}_img{img_index+1}.png",
                "is_high_res": width >= min_width
            })
    print(f" Extracted {len(image_data)} images")
    return image_data
def get_caption_candidates(text, image_data):
    print("Scanning for image captions...")
    lines = text.split("\n")
    results = []

    for img in image_data:
        context = {
            "page": img["page"],
            "image": img["filename"],
            "matched_caption": "",
            "has_citation": False,
            "has_building_name": False,
            "has_interior_note": False
        }
        for i, line in enumerate(lines):
            if f"{img['filename'].split('.')[0]}" in line:
                nearby_lines = lines[max(i-2, 0): i+3]
                caption_text = " ".join(nearby_lines)
                context["matched_caption"] = caption_text
                context["has_citation"] = any(x in caption_text.lower() for x in ["source", "http", "photographer"])
                context["has_building_name"] = bool(re.search(r"(building|tower|museum|villa|house|center)", caption_text, re.IGNORECASE))
                context["has_interior_note"] = bool(re.search(r"(interior|lobby|hall|inside)", caption_text, re.IGNORECASE))
                break
        results.append(context)
    return results
def evaluate_images_with_gemini(image_data, architect_name, debug=False):
    print(" Evaluating image content and relevance using Gemini...")
    enriched_image_feedback = []

    for img in tqdm(image_data, desc="Evaluating images"):
        prompt = f"""
You are reviewing an image submitted for a university architecture project about {architect_name}.
Please analyze this image and answer:

1. Does this image show a building designed by {architect_name}? If yes, specify the building.
2. Is this an interior or exterior shot?
3. Does this image clearly show architectural features (e.g., lighting, geometry, layout)?
4. How relevant is this image for an academic project about {architect_name}?

Give your feedback in the following JSON format:
{{
  "building_detected": "...",
  "interior_or_exterior": "...",
  "relevance_score": "x/10",
  "justification": "...",
  "architectural_features_visible": true/false
}}
"""
        try:
            response = vision_model.generate_content([img["image"], prompt])
            if debug:
                print(f"Image {img['filename']} feedback:\n", response.text)
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"```(?:json)?", "", cleaned_text)
                cleaned_text = cleaned_text.replace("```", "").strip()

            try:
                data = json.loads(cleaned_text)
            except Exception as e:
                print(f" Still failed to parse JSON from {img['filename']}: {e}")
                data = {
                    "building_detected": "Unknown",
                    "interior_or_exterior": "Unknown",
                    "relevance_score": "5/10",
                    "justification": "Could not parse feedback from Gemini.",
                    "architectural_features_visible": False
                }
        except Exception as e:
            print(f"⚠️ Error processing {img['filename']}: {e}")
            data = {
                "building_detected": "Unknown",
                "interior_or_exterior": "Unknown",
                "relevance_score": "5/10",
                "justification": "Could not extract structured feedback.",
                "architectural_features_visible": False
            }

        data.update({
            "filename": img["filename"],
            "page": img["page"],
            "width": img["width"],
            "height": img["height"],
            "is_high_res": img["is_high_res"]
        })
        enriched_image_feedback.append(data)
    return enriched_image_feedback
def evaluate_image_structure_and_captions(image_feedback, caption_context):
    print("Evaluating caption presence and structure")

    scores = []
    per_image_feedback = []

    for img in image_feedback:
        caption = next((c for c in caption_context if c["image"] == img["filename"]), {})
        has_citation = caption.get("has_citation", False)
        has_building_name = caption.get("has_building_name", False)
        has_interior_note = caption.get("has_interior_note", False)

        score = 0
        if has_citation: score += 3
        if has_building_name: score += 3
        if has_interior_note: score += 2
        if img["is_high_res"]: score += 2

        scores.append(score)
        per_image_feedback.append({
            "image": img["filename"],
            "page": img["page"],
            "relevance_score": img.get("relevance_score", "5/10"),
            "justification": img.get("justification", ""),
            "caption_found": caption.get("matched_caption", ""),
            "has_proper_caption": score >= 7,
            "score": score
        })

    avg_score = sum(scores) / len(scores) if scores else 0
    return {
        "score": int((avg_score / 10) * rubric["image_citations"]),
        "details": per_image_feedback
    }
def gemini_detailed_rubric_eval(text, architect_name, pdf_path):
    print(" Gemini evaluating full rubric with explanations")

    prompt = f"""
You are evaluating a student's architecture assignment on the architect {architect_name}.

This is a formal submission for university credit. You are receiving the full document as **images**, so you can directly observe the formatting, embedded images, captions, structure, and layout.

---

###  How to Grade:

- Be **fair and constructive**. If formatting is inconsistent, information is missing, or citations are weak, **please call it out clearly**.
- Do not sugarcoat — students are expected to revise based on your feedback.
- If something is strong, note it. If it's flawed, critique it.
- When scoring, **prioritize**:
  - Accuracy of academic citations
  - Caption and image attribution clarity
  - Clear distinction between interior vs exterior images
  - Overall layout and visual professionalism

---

###  Additional Clarifications:

-  Images are embedded (not just links)
-  Captions below images include attribution (URLs or photographer names)
-  A student photo and bio appear on Page 2
-  Table of Contents is present
-  10 buildings are described
-  Redundant links are likely citations, not missing content

---

###  RUBRIC CRITERIA

Please assess each of the following categories. For every criterion:

1. Give a **detailed justification** (1–2 paragraphs)
2. Assign a score **out of 5** based on the detailed rubric below

Format:
**[Category Name]**
Justification: ...
Score: x/5

---

###  Categories and Rubric Anchors

**1. Architect Selection & Scope**
- 5 = Clearly identifies one architect from Book Two, explicitly stated, on-topic
- 3–4 = Identifies Book Two architect, but clarity or justification could improve
- 1–2 = Architect unclear, off-topic, or not from Book Two

**2. Organization & Document Setup**
- 5 = Clear Table of Contents + labeled sections for bio, buildings, refs, student bio
- 3–4 = Minor issues with layout or missing headers
- 1–2 = Poor organization, missing sections, or hard to follow

**3. Biographical Content (750 words)**
- 5 = Covers who they are, achievements, education, significance, 1st building, typologies
- 3–4 = Mostly complete, with slight omissions or light detail
- 1–2 = Underdeveloped or below word count, missing major points

**4. Citation of Architect Biography**
- 5 = 5–10 academic references, correct APA formatting, includes DOIs and citation counts
- 3–4 = APA errors or missing DOIs, but still academically relevant
- 1–2 = Few or no academic references, poor or irrelevant sources

**5. Selection & Quality of Images**
- 5 = 10 buildings, 3+ exterior + 5+ interior per building, high-res
- 3–4 = Most buildings meet criteria; a few lack resolution or quantity
- 1–2 = Many buildings missing images or poor quality

**6. Image Citation & Attribution**
- 5 = Every image has clear, consistent source or photographer citation
- 3–4 = Most are cited but with some inconsistencies
- 1–2 = Citations mostly missing, inconsistent, or improperly formatted

**7. Coverage of 10 Famous Buildings**
- 5 = All 10 named + location + significance statement (1–2 sentences)
- 3–4 = Buildings listed but some lack significance or location
- 1–2 = Several missing or incomplete building descriptions

**8. Image Relevance**
- 5 = All images relate directly to described buildings, match descriptions, show architectural value
- 3–4 = Most images relevant, some generic or misaligned
- 1–2 = Several images are off-topic or not associated with described buildings

**9. Personal Bio & Photo**
- 5 = Professional photo and bio (1–2 sentences), correctly placed after TOC
- 3–4 = Present but minor formatting/image issues
- 1–2 = Photo or bio is low quality, misplaced, or absent

**10. Overall Completeness & Presentation**
- 5 = Fully polished, clean layout, minimal repetition, suitable for web/publication
- 3–4 = Clear submission, but lacks design polish or has formatting repetition
- 1–2 = Sloppy or rushed presentation; visual issues hurt readability

---

 Please start your rubric-based analysis below:
"""

    doc = fitz.open(pdf_path)
    all_pages_as_images = [page.get_pixmap(dpi=300).pil_tobytes("png") for page in doc]

    try:
        response = vision_model.generate_content(
             [prompt] + [Image.open(BytesIO(img_bytes)) for img_bytes in all_pages_as_images]
        )
    except Exception as e:
        print(f"Gemini Vision rubric evaluation failed: {e}")
        return {k: {"score": 0} for k in rubric.keys()}, ""  # Default to zeros to prevent crash

    print(response.text)

    def extract_score(label, out_of):
        # Try multiple patterns to extract the score
        patterns = [
            rf"{label}.*?Score:\s*(\d+)/{out_of}",  # Standard format
            rf"{label}.*?(\d+)/{out_of}",  # Just the score
            rf"{label}.*?Score:\s*(\d+)",  # Score without denominator
            rf"{label}.*?(\d+)\s*/\s*{out_of}"  # Score with spaces
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.text, re.IGNORECASE | re.DOTALL)
            if match:
                return int(match.group(1))
        
        # If not found in the main text, try to find it in the summary section
        summary_match = re.search(r"\*\*FINAL SUMMARY\*\*.*?(?=\*\*OVERALL COMMENTS|\Z)", response.text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary_text = summary_match.group(0)
            for pattern in patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE | re.DOTALL)
                if match:
                    return int(match.group(1))
        
        return 0

    scores = {
        "architect_chosen": {"score": extract_score("Architect Selection", 5)},
        "doc_and_slides": {"score": extract_score("Organization", 5)},
        "bio_750_words": {"score": extract_score("Biographical Content", 5)},
        "bio_structure": {"score": extract_score("Biographical Structure", 5)},
        "bio_references": {"score": extract_score("Citation of Architect Biography", 5)},
        "10_buildings_with_images": {"score": extract_score("Coverage of 10 Famous Buildings", 5)},
        "image_quality": {"score": extract_score("Selection & Quality of Images", 5)},
        "image_citations": {"score": extract_score("Image Citation & Attribution", 5)},
        "image_relevance": {"score": extract_score("Image Relevance", 5)},
        "personal_bio_photo": {"score": extract_score("Personal Bio", 5)},
        "presentation_polish": {"score": extract_score("Overall Completeness", 5)},
        "overall_completeness": {"score": extract_score("Overall Completeness", 5)}  # Add this for the admin interface
    }

    detailed_evaluation_text = response.text

    return scores, detailed_evaluation_text
def generate_detailed_scorecard(scores, image_caption_details=None):
    print(" Compiling final scorecard")

    # Total and max only for defined rubric keys
    total = sum([scores[k]["score"] for k in scores if k in rubric])
    max_total = sum([rubric[k] for k in scores if k in rubric])
    final_percentage = (total / max_total) * 100 if max_total else 0

    grade = "A" if final_percentage >= 50 else "B" if final_percentage >= 46 else "C" if final_percentage >= 42 else "D"

    # print(f"Final Grade: {grade} ({round(final_percentage, 2)}%)")
    rubric_table = pd.DataFrame([
        {
            "Criterion": k.replace("_", " ").title(),
            "Score": scores[k]["score"],
            "Max": rubric[k],
            "Description": rubric_descriptions.get(k, "")
        }
        for k in rubric if k in scores
    ])
    display(rubric_table)
    if image_caption_details:
        print("\n Image Caption & Relevance Feedback:")
        df = pd.DataFrame(image_caption_details["details"])
        display(df)

    return {
        "rubric_scores": {k: scores[k]["score"] for k in rubric if k in scores},
        "final_percent": round(final_percentage, 2),
        "grade": grade,
        "image_feedback_table": image_caption_details
    }
def extract_references_from_text(text):
    print(" Extracting references from text")
    lines = text.split("\n")
    references = []
    for line in lines:
        if re.search(r"\(\d{4}\)", line) and any(x in line.lower() for x in ["doi", "archdaily", "e-architect", "https://", "http://"]):
            references.append(line.strip())
    return references

def evaluate_biography(text):
    print(" Evaluating biography: checking word count and required sections")
    result = {}
    doc = nlp(text)
    result["word_count"] = len([token.text for token in doc if token.is_alpha])

    required_sections = [
        "who they are",
        "famous for",
        "studied",
        "significance",
        "influence",
        "types of buildings",
        "first building"
    ]

    section_hits = sum([1 for section in required_sections if section.lower() in text.lower()])
    result["structure_score"] = int((section_hits / len(required_sections)) * rubric["bio_structure"])
    result["score"] = rubric["bio_750_words"] if result["word_count"] >= 700 else int((result["word_count"] / 750) * rubric["bio_750_words"])

    return result

def evaluate_image_quality(image_data):
    print(" Evaluating image resolution")
    high_res_count = sum(1 for img in image_data if img["is_high_res"])
    total_images = len(image_data)
    
    quality_score = int((high_res_count / max(1, total_images)) * rubric["image_quality"])
    
    print(f" {high_res_count}/{total_images} images are high resolution")
    return {"high_res_count": high_res_count, "score": quality_score}

def run_autograder_full(pdf_path, architect_name="Bjarke Ingels", debug=False):
    print("Starting full grading pipeline")
    text = extract_text_from_pdf(pdf_path)
    
    # Get the scores and detailed evaluation from gemini_detailed_rubric_eval
    gemini_scores, detailed_evaluation_text = gemini_detailed_rubric_eval(text, architect_name, pdf_path)
    
    # Extract summary scores from the detailed evaluation
    summary_scores = {}
    summary_pattern = r"\*\*Final Summary:\*\*\s*(.*?)(?=\n\n|$)"
    summary_match = re.search(summary_pattern, detailed_evaluation_text, re.DOTALL)
    
    if summary_match:
        summary_text = summary_match.group(1)
        score_pattern = r"(\d+)\.\s+([^:]+):\s+(\d+)/5"
        matches = re.findall(score_pattern, summary_text)
        
        for _, category, score in matches:
            score = int(score)
            if "Architect Selection" in category:
                summary_scores["architect_chosen"] = score
            elif "Organization" in category:
                summary_scores["doc_and_slides"] = score
            elif "Biographical Content" in category:
                summary_scores["bio_750_words"] = score
            elif "Citation of Architect Biography" in category:
                summary_scores["bio_references"] = score
            elif "Selection & Quality of Images" in category:
                summary_scores["image_quality"] = score
            elif "Image Citation" in category:
                summary_scores["image_citations"] = score
            elif "Coverage of 10 Famous Buildings" in category:
                summary_scores["10_buildings_with_images"] = score
            elif "Image Relevance" in category:
                summary_scores["image_relevance"] = score
            elif "Personal Bio" in category:
                summary_scores["personal_bio_photo"] = score
            elif "Overall Completeness" in category:
                summary_scores["overall_completeness"] = score
                summary_scores["presentation_polish"] = score  # Keep both for backward compatibility
    
    # Use summary scores if available, otherwise fall back to extracted scores
    if summary_scores:
        print("Using summary scores for grade calculation")
        scores_to_use = summary_scores
    else:
        print("Using extracted scores for grade calculation")
        scores_to_use = {k: v["score"] for k, v in gemini_scores.items()}
    
    # Calculate total score out of 50 (10 criteria × 5 points each)
    total = sum(scores_to_use.values())
    max_total = 50  # 10 criteria × 5 points each
    final_percent = round((total / max_total) * 100, 2)
    
    # More granular grade calculation based on percentage
    if final_percent >= 93:
        grade = "A"
    elif final_percent >= 90:
        grade = "A-"
    elif final_percent >= 87:
        grade = "B+"
    elif final_percent >= 83:
        grade = "B"
    elif final_percent >= 80:
        grade = "B-"
    elif final_percent >= 77:
        grade = "C+"
    elif final_percent >= 73:
        grade = "C"
    elif final_percent >= 70:
        grade = "C-"
    elif final_percent >= 67:
        grade = "D+"
    elif final_percent >= 63:
        grade = "D"
    elif final_percent >= 60:
        grade = "D-"
    else:
        grade = "F"

    # Print the scores and grade for debugging
    print(f"Total score: {total}/{max_total} = {final_percent}%")
    print(f"Grade: {grade}")
    print("Scores used for calculation:")
    for key, value in scores_to_use.items():
        print(f"  {key}: {value}/5")

    return {
        "rubric_scores": scores_to_use,
        "final_percent": final_percent,
        "grade": grade,
        "detailed_evaluation": detailed_evaluation_text
    }

if __name__ == "__main__":
    result = run_autograder_full("/path/to/sample.pdf", "Bjarke Ingels", debug=True)
