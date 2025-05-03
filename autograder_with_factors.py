import os
import re
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import spacy
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load models and API keys
nlp = spacy.load("en_core_web_sm")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel("gemini-2.0-flash")

# 1) Factorized rubric definition
rubric_factors = {
    "Architect Selection & Scope": {
        1: [  # Exemplary
            "Clearly names one architect from Book Two",
            "Selection fully adheres to course requirements",
            "Architect explicitly stated in the document"
        ],
        2: [  # Good
            "Names an architect from Book Two",
            "Minor details or justification lacking but meets requirement"
        ],
        3: [  # Satisfactory
            "Identifies an architect with some ambiguity or misalignment"
        ],
        4: [  # Needs Improvement
            "Fails to clearly identify a Book Two architect",
            "Selection is off-scope or missing"
        ]
    },
    "Organization & Document Setup": {
        1: [
            "Includes a clear Table of Contents",
            "Has 'architect background' section",
            "Has '10 buildings' section",
            "Has 'academic references' section",
            "Has 'personal bio' section",
            "Layout follows recommended doc structure"
        ],
        2: [
            "Includes most required sections; minor heading/order issues"
        ],
        3: [
            "Includes sections but not clearly distinguished or organized"
        ],
        4: [
            "Poorly organized; critical sections missing or hard to identify"
        ]
    },
    "Biographical Content (750 words)": {
        1: [
            "Contains >=750-word biography",
            "Covers identity (who the architect is)",
            "Covers achievements",
            "Covers education",
            "Covers historical significance",
            "Covers first attributed building",
            "Covers types of buildings",
            "Supported by academic citations"
        ],
        2: [
            "Biography approx 750 words",
            "Covers main topics; minor omissions"
        ],
        3: [
            "Biography present but underdeveloped or <750 words",
            "Lacks sufficient detail in one or more areas"
        ],
        4: [
            "Incomplete or significantly under <750 words",
            "Off-topic or many key elements missing"
        ]
    },
    "Citation of Architect Biography": {
        1: [
            "5-10 refs in correct APA format",
            "Citations include DOIs and citation counts",
            "Every claim supported by sources"
        ],
        2: [
            ">=5 refs in APA; minor formatting issues",
            "Most citations include DOIs and are appropriate"
        ],
        3: [
            "<5 refs or multiple APA errors",
            "Some refs may lack credibility"
        ],
        4: [
            "Minimal or no refs",
            "Citations largely incorrect or irrelevant"
        ]
    },
    "Selection & Quality of Images": {
        1: [
            "Each building has >=3 high-res exterior images",
            "Each building has >=5 high-res interior images",
            "Images demonstrate building features"
        ],
        2: [
            "Most buildings include required number of high-res images",
            "Images meet quality standards with few exceptions"
        ],
        3: [
            "Some buildings have insufficient or low-quality images",
            "Selection uneven"
        ],
        4: [
            "Fails to provide required number or quality for most buildings",
            "Many images missing or poorly chosen"
        ]
    },
    "Image Citation & Attribution": {
        1: [
            "Every image has clear, accurate citation (photographer/source)"
        ],
        2: [
            "Most images properly cited; minor errors"
        ],
        3: [
            "Some images cited, many missing; inconsistency present"
        ],
        4: [
            "Majority lack proper citation; citations incorrect"
        ]
    },
    "Coverage of 10 Famous Buildings": {
        1: [
            "Details for 10 buildings: name, location, 1-2 sentence significance",
            "Image suggestions provided consistently"
        ],
        2: [
            "All 10 buildings covered with essential details; some less detailed"
        ],
        3: [
            "Fewer than 10 or some entries lack significance or image details"
        ],
        4: [
            "Covers <10 buildings; information largely missing or incorrect"
        ]
    },
    "Personal Bio & Photo": {
        1: [
            "Professional bio page with high-res photo",
            "1-2 sentence bio placed correctly"
        ],
        2: [
            "Bio and photo included; minor placement or quality issues"
        ],
        3: [
            "Bio minimal or photo low quality; placement off"
        ],
        4: [
            "Bio/photo missing or do not meet requirements"
        ]
    },
    "Overall Completeness & Presentation": {
        1: [
            "Doc and slides complete, polished, professional",
            "All requirements met; suitable for web posting"
        ],
        2: [
            "Overall solid; minor formatting or content issues"
        ],
        3: [
            "Meets basic requirements; several formatting/clarity issues"
        ],
        4: [
            "Major sections incomplete or poorly formatted; errors affect clarity"
        ]
    }
}

def check_factors(text: str, criterion: str) -> dict:
    """
    Returns a dict mapping (criterion, col, idx) → bool indicating whether each factor passed.
    """
    results = {}
    content = text.lower()
    for col, factors in rubric_factors[criterion].items():
        for idx, desc in enumerate(factors, start=1):
            key = (criterion, col, idx)
            # simple substring check; you can refine with regex if needed
            results[key] = desc.lower() in content
    return results

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def run_autograder_with_factors(pdf_path: str, architect_name: str):
    text = extract_text(pdf_path)
    # 4a) Collect factor passes for each criterion
    factor_results = {}
    for criterion in rubric_factors:
        factor_results[criterion] = check_factors(text, criterion)
    # 4b) Build factor table for True results
    factor_table = []
    for criterion, checks in factor_results.items():
        row = { 'Criterion': criterion }
        # columns 1-4
        for col in range(1,5):
            passed = [f"({criterion}, {col}, {idx})"
                      for (crit, c, idx), ok in checks.items()
                      if crit == criterion and c == col and ok]
            row[f'Col_{col}'] = ', '.join(passed) or '-'
        factor_table.append(row)
    df_factors = pd.DataFrame(factor_table)
    # 5) Reflective prompt to LLM
    reflective_prompt = (
        "I have graded a submission using the following factor table:\n" +
        df_factors.to_markdown(index=False) +
        "\n\nPlease verify whether these factor checks align with the rubric definitions, and suggest any corrections."
    )
    reflect = text_model.generate_content([reflective_prompt]).text
    return {
        "factor_table": df_factors,
        "reflection": reflect
    } 