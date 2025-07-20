import streamlit as st
import requests
import base64
import pandas as pd
import re
from io import BytesIO
import json
import os
import io
import pytesseract
from PIL import Image, ImageEnhance, ImageChops, ImageFilter, ImageOps
import numpy as np
import cv2
import fitz
# Remove PaddleOCR import
# from paddleocr import PaddleOCR
from skimage.restoration import denoise_nl_means #scikit-image for enhancement for images
from skimage import img_as_float
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv


# Remove PaddleOCR initialization
# ocr_engine = PaddleOCR(
#     use_angle_cls=True,
#     lang='en'
# )

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env file")
# Use Gemini 1.5 Flash endpoint for best speed
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

def gemini_chat_completion(messages, temperature=0.1):
    # messages: list of dicts with 'role' and 'content' (text or image)
    # Only text and image_url supported
    contents = []
    for msg in messages:
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part.get("type") == "text":
                    contents.append({"role": msg["role"], "parts": [{"text": part["text"]}]})
                elif part.get("type") == "image_url":
                    contents.append({"role": msg["role"], "parts": [{"inlineData": {"mimeType": "image/png", "data": part["image_url"]["url"].split(",")[-1]}}]})
        else:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    payload = {"contents": contents}
    response = requests.post(GEMINI_API_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    # Gemini returns candidates[0].content.parts[0].text
    return result["candidates"][0]["content"]["parts"][0]["text"]


# st.set_page_config(page_title="NeoICR - Vision Powered Invoice Extractor", layout="wide")


def clean_and_parse_json(raw_text):
    """
    Removes markdown-style backticks and parses the text into JSON.
    """
    try:
        cleaned_text = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        parsed_data = json.loads(cleaned_text)
        return parsed_data, "Valid"
    except Exception:
        return raw_text, "Invalid Format"

def run_paddle_ocr(img: Image.Image):
    try:
        # Use pytesseract for OCR
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print("OCR processing failed:", str(e))
        return ""

def correct_image_rotation(img):
    try:
        ocr_data = pytesseract.image_to_osd(img)
        ocr_lines = ocr_data.split("\n")
        if len(ocr_lines) >= 3:
            rotation_line = ocr_lines[2]
            rotation = int(rotation_line.split(":")[-1].strip()) 
            if rotation != 0:
                img = img.rotate(-rotation, expand=True)
        else:
            print(f"Warning: OCR data doesn't have enough lines for rotation. OCR data: {ocr_data}")
    except Exception as e:
        print(f"Error during OCR-based rotation correction: {e}")
    return img

def crop_image(img, background_color=(255, 255, 255)):
    bg = Image.new(img.mode, img.size, background_color)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img

def enhance_image(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Adjust contrast

    sharpener = ImageEnhance.Sharpness(img)
    img = sharpener.enhance(1.5)  # Adjust sharpness
    return img

def bilateral_filter(img, diameter=9, sigma_color=75, sigma_space=75):
    img_np = np.array(img)
    img_filtered = cv2.bilateralFilter(img_np, diameter, sigma_color, sigma_space)
    return Image.fromarray(img_filtered)

def denoise_image(img):
    img = img.convert('L')  # Convert to grayscale for denoising
    img_array = img_as_float(np.array(img))  # Convert to float representation
    denoised_img_array = denoise_nl_means(img_array, patch_size=5, patch_distance=6)
    denoised_img = Image.fromarray((denoised_img_array * 255).astype(np.uint8))
    return denoised_img

def binarize_image(img: Image.Image) -> Image.Image:
    img = img.convert('L')
    gray = np.array(img)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return Image.fromarray(binary).convert("L")

def preprocess_image(img):
    img = correct_image_rotation(img)  
    img = crop_image(img)  
    img = bilateral_filter(img) 
    img = denoise_image(img)   
    img = enhance_image(img)     
    img = binarize_image(img) 
    return img

def convert_images_to_base64(images):
    def process(img):
        print("Preprocessing image...")
        img = preprocess_image(img)
        print("Image preprocessed")

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    with ThreadPoolExecutor() as executor:
        base64_images = list(executor.map(process, images))
    return base64_images

def extract_data_from_images(images_base64):

    system_prompt = """You are an expert invoice data extractor. Extract information accurately 
    and maintain context between pages. Be extremely thorough with product listings - 
    ensure you capture EVERY single line item visible on the invoice. If two product lines appear close together or are not separated by borders,
    treat each visible row as a distinct product entry unless it's clearly a wrapped description.
    Check especially the top and bottom of the page for any partial or wrapped line items that may be split across pages.
    If a line item spans multiple lines, merge them unless there’s a clear delimiter. Always treat visible horizontal alignment as row boundary."""

    extraction_prompt_template = """
    You are analyzing an invoice image to extract key structured data. You are also provided with a raw OCR text output from the image.
    
    First, look at the image **visually** to understand its content. Then, compare the visual content with the **raw OCR output**.

    If there are any discrepancies, always trust what you visually see in the image **unless the text is blurry or unclear** — in which case, cautiously fall back on the OCR result if it appears sensible.

    Extract the following fields from the invoice image (based on the more accurate source):

    - **Inv No**: Look specifically for labels such as "GST Inv No.","Invoice No.", "Invoice Number", "Bill No.", "Reference No." or similar terms. Extract the exact value from invoice.
    - **Invoice Date**: Date of the invoice.
    - **Invoice Value**: Total value of the invoice.
    - **Eway Bill No**: Eway bill number present on the invoice. Look for labels like "EWB No.".
    - **Eway Bill Date**: Eway bill date present on the invoice.
    - **Eway Bill Expiry Date**: Eway bill expiry date present on the invoice.

    - **Products**: Extract every line item from the invoice section in this image. Include for each product:
        - **Part No** : Always starts with "98"
        - **Product Desc** : Full product description. Include all specifications, models, part numbers, etc.(Incase of separate columns of product and descript, merge them)
        - **Part Quantity**
        - **Weight**
        - **Charged Weight**
        - **Units**
        - **Length** : If not found, default to 0.
        - **Width** : If not found, default to 0.
        - **Height** If not found, default to 0.

    Return a JSON with the metadata fields at the root level, and the list of product objects in a "Products" array.

    IMPORTANT: This is page {page_number} of a {total_pages}-page document. This may be a completely separate invoice from other pages. Treat each page as a potential separate invoice with its own invoice number. Use raw OCR **only as a secondary reference**.
    """

    batch_results = []
    total_pages = len(images_base64)

    for i, img_data in enumerate(images_base64):
        try:
            page_number = i + 1
            extraction_prompt = extraction_prompt_template.format(
                page_number=page_number, total_pages=total_pages
            )

            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))

            try:
                ocr_text = run_paddle_ocr(img)
                print(ocr_text)
            except Exception as e:
                print(f"OCR processing failed: {e}")
                ocr_text = ""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_prompt + "\n\nRaw OCR Text:\n" + ocr_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                    ]
                }
            ]

            response = gemini_chat_completion(messages)

            batch_results.append(response)

        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            continue

    consolidated_result = consolidate_results(batch_results)
    return json.dumps(consolidated_result)


def consolidate_results(batch_results):
    if not batch_results:
        return {"error": "No results to consolidate"}
        
    consolidation_prompt = """
    You are given extracted data from different pages of a PDF document. Each page might represent a completely separate invoice.
    
    Key rules:
    1. Group products by their associated invoice number
    2. If the invoice number is the same across multiple pages, those pages belong to the same invoice
    3. If the invoice number changes, it indicates a new invoice within the same PDF
    4. Ensure each invoice has its complete metadata and product list
    5. Return an array of invoices, each containing its metadata and products
    Here is the data from different pages:
    {batch_data}
    
    Consolidate and return a JSON array of invoice objects, where each invoice has:
    1. Invoice metadata (Inv No, Invoice Date etc.)
    2. A Products array containing all products for that invoice
    """
    
    batch_data_str = json.dumps(batch_results, indent=2)
    
    response = gemini_chat_completion(
        [{"role": "system", "content": "You are an expert invoice data consolidator."},
         {"role": "user", "content": consolidation_prompt.format(batch_data=batch_data_str)}]
    )
    
    try:
        result = response
        if "```json" in result:
            json_str = result.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        else:
            return json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse consolidated result", "raw": result}


def clean_json_response(raw_response):
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(json_pattern, raw_response)
    json_str = match.group(1).strip() if match else raw_response.strip()
    return json_str.replace('\n', ' ')

def process_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        images = []
        for page in doc[1:]: 
            pix = page.get_pixmap(dpi=400)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)

        images_base64 = convert_images_to_base64(images)
        json_result = extract_data_from_images(images_base64)

        try:
            return {"filename": pdf_file.name, "data": json.loads(json_result)}
        except json.JSONDecodeError:
            print("JSON Decode Error, trying to clean the response")
            try:
                cleaned_json = clean_json_response(json_result)
                return {"filename": pdf_file.name, "data": json.loads(cleaned_json)}
            except json.JSONDecodeError:
                return {"filename": pdf_file.name, "error": "Failed to parse JSON", "raw_response": json_result}
    except Exception as e:
        return {"filename": pdf_file.name, "error": str(e), "raw_response": str(e)}


def prepare_data_for_dataframe(results):
    all_data = []

    for result in results:
        if "error" in result:
            all_data.append({"Filename": result["filename"], "Error": result["error"], "Status": "Error"})
            continue

        extracted_data = result.get("data", [])
        if not isinstance(extracted_data, list):
            extracted_data = [extracted_data]

        for invoice_data in extracted_data:
            invoice_metadata = {
                "Filename": result["filename"],
                "Invoice_Number": invoice_data.get("Inv No", "None"),
                "Invoice Date": invoice_data.get("Invoice Date", "None"),
                "Invoice Value": invoice_data.get("Invoice Value", "None"),
                "Eway Bill No": invoice_data.get("Eway Bill No", "None"),
                "Eway Bill Date": invoice_data.get("Eway Bill Date", "None"),
                "Eway Bill Expiry Date": invoice_data.get("Eway Bill Expiry Date", "None"),
            }

            products = invoice_data.get("Products", [])
            has_valid_product = False

            for product in products:
                if isinstance(product, dict):
                    product_desc = str(product.get("Product Desc", "")).strip().lower()
                    if product_desc and product_desc != "none" and product_desc!= "Unknown":
                        has_valid_product = True
                        row = invoice_metadata.copy()
                        row.update({
                            "Part No": product.get("Part No", "None"),
                            "Product Desc": product.get("Product Desc", "None"),
                            "Part Quantity": product.get("Part Quantity", "None"),
                            "Weight": product.get("Weight", "None"),
                            "Charged Weight": product.get("Charged Weight", "None"),
                            "Units": product.get("Units", "None"),
                            "Length": product.get("Length", "None"),
                            "Width": product.get("Width", "None"),
                            "Height": product.get("Height", "None"),
                        })
                        all_data.append(row)

            if not products or not has_valid_product:
                continue 

    return all_data




def main():
    st.title("NeoICR - Invoice Extractor")

    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results' not in st.session_state:
        st.session_state.results = []

    uploaded_files = st.file_uploader("Upload PDF invoices", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process Invoices"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, pdf_file in enumerate(uploaded_files):
                status_text.text(f"Processing {pdf_file.name} ({i+1}/{len(uploaded_files)})")
                result = process_pdf(pdf_file)
                results.append(result)
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Processing complete!")
            st.session_state.results = results
            st.session_state.processed = True


    if st.session_state.processed and st.session_state.results:
        results = st.session_state.results
        error_results = [r for r in results if "error" in r]

        try:
            all_data = prepare_data_for_dataframe(results)

            if all_data:
                df = pd.DataFrame(all_data)
                st.success(f"Successfully prepared {len(df)} rows")
                st.header("Extracted Data")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="invoice_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("No data extracted")
        except Exception as e:
            st.error(f"Error creating DataFrame: {e}")

        if error_results:
            st.header("Errors")
            for error in error_results:
                st.error(f"{error['filename']}: {error['error']}")
                if "raw_response" in error:
                    with st.expander(f"Raw Output for {error['filename']}"):
                        st.text(error["raw_response"])

if __name__ == "__main__":
    main()
