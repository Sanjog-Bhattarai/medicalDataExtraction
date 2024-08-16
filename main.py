from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import pytesseract
import re
import numpy as np
import cv2
from io import BytesIO
import json
import os

app = FastAPI()

# Set the path to the Tesseract executable (Adjust this path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path if necessary

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML form for file upload
@app.get("/", response_class=HTMLResponse)
async def main():
    with open("templates/upload.html", "r") as f:
        content = f.read()
    return content

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_text(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image)
    return text

def parse_text(text: str) -> dict:
    name_pattern_patient = re.compile(r"(?:Name of Patient|name of patient)[\s:]*([\w\s]+)", re.IGNORECASE)
    name_pattern = re.compile(r"(?:Name|name)[\s:]*([\w\s]+)", re.IGNORECASE)
    gender_pattern = re.compile(r"(?:Gender|GENDER|gender)[\s:]*([MFmf])")
    diagnosis_pattern = re.compile(
        r"(?:Disease|DISEASE|disease|Diagnosis|DIAGNOSIS|diagnosis|Medical Condition|MEDICAL CONDITION|medical condition|PRINCIPAL DIAGNOSIS|principal diagnosis|Principal Diagnosis)[\s:]*([^\n]*)",
        re.MULTILINE
    )
    age_pattern = re.compile(r"(?:Age|AGE|age)[\s:]*([\d]+)")
    dob_pattern = re.compile(
        r"(?:DOB|dob|Date of Birth|DATE OF BIRTH|Date Of Birth|DATE OF BIRTH)[\s:]*("
        r"\d{2}/\d{2}/\d{4}|"
        r"\d{2}/\d{2}/\d{4}|"
        r"[A-Za-z]+\s\d{1,2},\s\d{4})",
        re.IGNORECASE
    )

    name_match_patient = name_pattern_patient.search(text)
    name_match = name_pattern.search(text)
    gender_match = gender_pattern.search(text)
    diagnosis_match = diagnosis_pattern.search(text)
    age_match = age_pattern.search(text)
    dob_match = dob_pattern.search(text)

    if name_match_patient:
        name = name_match_patient.group(1).strip()
    elif name_match:
        name = name_match.group(1).strip()
    else:
        name = "N/A"

    if name != "N/A":
        name_parts = name.split()
        if len(name_parts) >= 2:
            name = f"{name_parts[0]} {name_parts[1]}"
        else:
            name = "N/A"

    gender = gender_match.group(1).strip().upper() if gender_match else "N/A"
    if gender not in ["M", "F"]:
        gender = "N/A"

    return {
        "name": name,
        "gender": gender,
        "disease/diagnosis/medical_condition": diagnosis_match.group(1).strip() if diagnosis_match else "N/A",
        "age": age_match.group(1).strip() if age_match else "N/A",
        "dob": dob_match.group(1).strip() if dob_match else "N/A",
    }

def save_to_json(data: dict, filename: str):
    os.makedirs("extracted_data", exist_ok=True)
    filepath = os.path.join("extracted_data", filename)
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)

@app.post("/extract/", response_class=HTMLResponse)
async def extract_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents))
        rgb_image = pil_image.convert("RGB")
        image_array = np.array(rgb_image)
        preprocessed_image = preprocess_image(image_array)
        extracted_text = extract_text(preprocessed_image)
        extracted_data = parse_text(extracted_text)
        filename = f"{file.filename.split('.')[0]}.json"
        save_to_json(extracted_data, filename)

        # Create an HTML response with styled data
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                    padding: 20px;
                }}
                h1 {{
                    color: #444;
                }}
                .data-container {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    margin-top: 20px;
                }}
                .data-item {{
                    margin-bottom: 10px;
                }}
                .data-label {{
                    font-weight: bold;
                    color: #555;
                }}
            </style>
        </head>
        <body>
            <h1>Extracted Data</h1>
            <div class="data-container">
                <div class="data-item">
                    <span class="data-label">Name:</span> {extracted_data['name']}
                </div>
                <div class="data-item">
                    <span class="data-label">Gender:</span> {extracted_data['gender']}
                </div>
                <div class="data-item">
                    <span class="data-label">Disease/Diagnosis/Medical Condition:</span> {extracted_data['disease/diagnosis/medical_condition']}
                </div>
                <div class="data-item">
                    <span class="data-label">Age:</span> {extracted_data['age']}
                </div>
                <div class="data-item">
                    <span class="data-label">Date of Birth:</span> {extracted_data['dob']}
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
