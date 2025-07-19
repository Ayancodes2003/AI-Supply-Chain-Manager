import streamlit as st
import pandas as pd
import re
from io import BytesIO
import json
import os
import io
import pytesseract
from PIL import Image, ImageEnhance, ImageChops, ImageFilter, ImageOps, ImageDraw
import numpy as np
import cv2
import fitz
# Remove PaddleOCR initialization and get_paddle_ocr_engine
# ocr_engine = get_paddle_ocr_engine()
from skimage.restoration import denoise_nl_means
from skimage import img_as_float
from fuzzywuzzy import fuzz
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode

# This module is for Human-in-the-Loop (HITL) verification only. No LLM or OpenAI code is used here.
# st.set_page_config(page_title="NeoICR - Vision Powered Invoice Verification", layout="wide")


# Global OCR engine
# Remove PaddleOCR initialization and get_paddle_ocr_engine
# @st.cache_resource
# def get_paddle_ocr_engine():
#     return PaddleOCR(use_angle_cls=True, lang='en', structure=True, layout=True)

# ocr_engine = get_paddle_ocr_engine()


# Helper to convert list of points to axis-aligned bbox
def get_axis_aligned_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def preprocess_image(image: Image.Image):
    # Convert to grayscale
    img_gray = image.convert('L')

    # Convert to numpy array for OpenCV
    img_np = np.array(img_gray)

    # Denoising
    img_float = img_as_float(img_np)
    img_denoised = denoise_nl_means(img_float, h=0.1, fast_mode=True, patch_size=5, patch_distance=6)
    img_np = (img_denoised * 255).astype(np.uint8)

    # Deskewing
    try:
        osd_result = pytesseract.image_to_osd(img_np)
        angle_match = re.search(r"Rotate: (\d+)\nOrientation in degrees: (\d+)", osd_result)
        if angle_match:
            rotation_angle = int(angle_match.group(1))
            if rotation_angle != 0:
                img_pil_temp = Image.fromarray(img_np)
                img_pil_temp = img_pil_temp.rotate(-rotation_angle, expand=True)
                img_np = np.array(img_pil_temp)
    except Exception as e:
        pass # Suppress warning for cleaner UI if Tesseract is optional

    # Binarization (Otsu's method)
    _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_bin, -1, kernel)

    return Image.fromarray(img_sharpened)

def run_paddle_ocr_with_bboxes(img: Image.Image):
    # Use pytesseract to get bounding boxes and text
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    detailed_ocr_output = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
            bbox = [data['left'][i], data['top'][i], data['left'][i]+data['width'][i], data['top'][i]+data['height'][i]]
            detailed_ocr_output.append({
                "text": data['text'][i],
                "bbox": bbox,
                "type": "text_line"
            })
    return detailed_ocr_output

def process_pdf_for_ocr_data(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Store preprocessed images and their OCR results for each page
        pages_data = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=400)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            preprocessed_img = preprocess_image(img.copy())
            page_ocr_results = run_paddle_ocr_with_bboxes(preprocessed_img)

            pages_data.append({
                "page_num": i + 1,
                "image": preprocessed_img,
                "ocr_results": page_ocr_results
            })
        return {"filename": pdf_file.name, "pages": pages_data, "status": "Processed"}
    except Exception as e:
        return {"filename": pdf_file.name, "error": str(e), "status": "Error"}


# MODIFIED: Function to apply transparent fill and outline
def apply_highlight_to_image(image: Image.Image, bbox_coords: list, highlight_color="#FFFF00", outline_width=3, fill_opacity=0.4):
    if not bbox_coords or len(bbox_coords) != 4:
        return image

    try:
        x1, y1, x2, y2 = bbox_coords

        # Ensure x1,y1,x2,y2 are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        if x1 >= x2 or y1 >= y2: # Invalid box (width/height is zero or negative)
            return image

        # Convert image to RGBA if it's not already, necessary for alpha_composite
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create a transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Parse highlight_color (hex string) to RGB for the fill
        r = int(highlight_color[1:3], 16)
        g = int(highlight_color[3:5], 16)
        b = int(highlight_color[5:7], 16)
        fill_color_rgba = (r, g, b, int(255 * fill_opacity)) # Apply opacity to the fill color

        # Draw filled rectangle (highlight)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color_rgba)

        # Draw an outline (using the selected color with full opacity for visibility)
        for i in range(outline_width):
            # Adjust outline coordinates to ensure it's drawn within the image boundaries
            outline_x1 = max(0, x1 - i)
            outline_y1 = max(0, y1 - i)
            outline_x2 = min(image.width, x2 + i)
            outline_y2 = min(image.height, y2 + i)
            draw.rectangle([outline_x1, outline_y1, outline_x2, outline_y2], outline=highlight_color)


        # Composite the overlay onto the original image and convert back to RGB for display
        return Image.alpha_composite(image, overlay).convert('RGB')

    except Exception as e:
        st.warning(f"Could not draw highlight for bbox {bbox_coords}: {e}")
        return image

def find_best_ocr_match(search_text: str, ocr_results_for_pages: list):
    """
    Finds the best matching OCR item across all pages for a given search_text.
    Returns (best_page_num, best_bbox, best_score)
    """
    best_match_info = None
    best_score = -1

    if not search_text or search_text.strip() == "":
        return None, None, -1

    search_text_lower = str(search_text).lower() # Ensure search_text is a string

    for page_data in ocr_results_for_pages:
        page_num = page_data["page_num"]
        ocr_items = page_data["ocr_results"]
        for item in ocr_items:
            if item["text"] is not None and item["bbox"] is not None:
                # Use partial_ratio as extracted values might be parts of OCR text
                score = fuzz.partial_ratio(search_text_lower, str(item["text"]).lower())
                if score > best_score:
                    best_score = score
                    best_match_info = (page_num, item["bbox"])
    return best_match_info[0] if best_match_info else None, \
           best_match_info[1] if best_match_info else None, \
           best_score


def main():
    # Session state initialization
    if 'pdf_ocr_data' not in st.session_state:
        st.session_state.pdf_ocr_data = {} # Stores {filename: [{page_num, image, ocr_results}]}
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = pd.DataFrame()
    if 'aggrid_selected_cell_data' not in st.session_state: # To store selected cell data from AgGrid
        st.session_state.aggrid_selected_cell_data = None
    if 'current_display_image' not in st.session_state:
        st.session_state.current_display_image = None
    if 'current_display_caption' not in st.session_state:
        st.session_state.current_display_caption = "No document selected or data point chosen."
    if 'highlight_color' not in st.session_state:
        st.session_state.highlight_color = "#FFFF00" # Default to yellow
    if 'last_column_selection' not in st.session_state: # To persist dropdown selection
        st.session_state.last_column_selection = ''


    st.title("NeoICR - Invoice Verification & Visualizer")

    st.sidebar.header("1. Upload PDFs for OCR")
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDF invoices to generate OCR data", type="pdf", accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_pdfs and st.sidebar.button("Process PDFs for OCR Data", key="process_pdfs_button"):
        st.session_state.pdf_ocr_data = {} # Clear previous OCR data
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        for i, pdf_file in enumerate(uploaded_pdfs):
            status_text.text(f"Processing OCR for {pdf_file.name} ({i+1}/{len(uploaded_pdfs)})")
            result = process_pdf_for_ocr_data(pdf_file)
            if result["status"] == "Processed":
                st.session_state.pdf_ocr_data[result["filename"]] = result["pages"]
            else:
                st.error(f"Error processing {pdf_file.name}: {result['error']}")
            progress_bar.progress((i + 1) / len(uploaded_pdfs))
        status_text.success("OCR data generation complete!")
        st.sidebar.info(f"OCR data generated for {len(st.session_state.pdf_ocr_data)} PDFs.")

    st.sidebar.header("2. Upload Your Extracted CSV Data")
    uploaded_csv = st.sidebar.file_uploader(
        "Upload your CSV file with extracted invoice data (must contain a 'Filename' column matching PDFs)",
        type="csv", key="csv_uploader"
    )

    if uploaded_csv:
        try:
            st.session_state.uploaded_df = pd.read_csv(uploaded_csv)
            if "Filename" not in st.session_state.uploaded_df.columns:
                st.error("Error: The uploaded CSV must contain a 'Filename' column.")
                st.session_state.uploaded_df = pd.DataFrame() # Clear invalid df
            else:
                st.sidebar.success("CSV data loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.session_state.uploaded_df = pd.DataFrame() # Clear invalid df

    st.sidebar.header("3. Highlight Color Settings")
    st.session_state.highlight_color = st.sidebar.color_picker(
        "Choose Highlight Color",
        value=st.session_state.highlight_color,
        key="color_picker"
    )

    st.header("4. Your Extracted Data (from CSV)")

    if not st.session_state.uploaded_df.empty:
        st.write("Click on any cell in the table below to see its corresponding highlight on the PDF.")
        st.write("Or, use the dropdown below to highlight specific columns or 'All' columns for the *currently selected row*.")


        # --- AgGrid Setup ---
        gb = GridOptionsBuilder.from_dataframe(st.session_state.uploaded_df)
        gb.configure_selection(
            'single',
            use_checkbox=False,
            groupSelectsChildren=False,
            rowMultiSelectWithClick=False,
            suppressRowDeselection=True,
            enableRangeSelection=True
        )
        gb.configure_grid_options(domLayout='normal')
        gridOptions = gb.build()

        grid_response = AgGrid(
            st.session_state.uploaded_df,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.MODEL_CHANGED, # MODEL_CHANGED will trigger rerun on selection
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=True,
            height=350,
            width='100%',
            reload_data=True,
            key='aggrid_table',
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
        )

        # Check if a cell was newly selected in AgGrid
        # Or if a cell was previously selected and we're just changing the column filter via dropdown
        selected_row_data = None
        if grid_response['selected_cells']:
            selected_cell_data = grid_response['selected_cells'][0]
            st.session_state.aggrid_selected_cell_data = selected_cell_data
            selected_row_data = selected_cell_data['data'] # Get the full row data of the newly selected cell
        elif st.session_state.aggrid_selected_cell_data:
            # If no new cell clicked, but a cell was previously selected, use its row data
            selected_row_data = st.session_state.aggrid_selected_cell_data['data']


        # If a row (or cell within a row) is selected
        if selected_row_data:
            columns_for_selection = [col for col in st.session_state.uploaded_df.columns if col != "Filename"]

            # Determine default index for the selectbox
            default_index = 0 # Default to empty string ''
            if st.session_state.get('last_column_selection') == 'All':
                default_index = len(columns_for_selection) + 1 # Index for 'All'
            elif st.session_state.aggrid_selected_cell_data and st.session_state.aggrid_selected_cell_data['colId'] in columns_for_selection:
                # If a cell was just clicked, default the dropdown to that column
                default_index = columns_for_selection.index(st.session_state.aggrid_selected_cell_data['colId']) + 1 # +1 for ''
            elif st.session_state.get('last_column_selection') in columns_for_selection:
                # If dropdown was previously set to a specific column, maintain it
                default_index = columns_for_selection.index(st.session_state.last_column_selection) + 1


            st.session_state.selected_column_name_dropdown = st.selectbox(
                "Select a column to highlight for the currently selected row (or 'All'):",
                options=[''] + columns_for_selection + ['All'], # Added 'All' option
                index=default_index,
                key="column_selector_dropdown"
            )
            # Update last selected column for next rerun
            st.session_state.last_column_selection = st.session_state.selected_column_name_dropdown


            # Determine the value(s) to highlight based on dropdown selection
            value_to_highlight_list = []
            selected_display_name = ""

            # Prioritize dropdown selection if active, otherwise use AgGrid click
            if st.session_state.selected_column_name_dropdown == 'All':
                # Iterate through all columns in the selected row (excluding Filename)
                for col_name, val in selected_row_data.items():
                    if col_name != "Filename" and pd.notna(val) and str(val).strip() != "":
                        value_to_highlight_list.append({"column": col_name, "value": str(val)})
                selected_display_name = "All relevant fields"
            elif st.session_state.selected_column_name_dropdown: # Specific column selected from dropdown
                col_name = st.session_state.selected_column_name_dropdown
                val = selected_row_data.get(col_name)
                if pd.notna(val) and str(val).strip() != "":
                    value_to_highlight_list.append({"column": col_name, "value": str(val)})
                selected_display_name = f"'{col_name}'"
            elif st.session_state.aggrid_selected_cell_data: # Fallback to initial AgGrid cell click
                col_name = st.session_state.aggrid_selected_cell_data['colId']
                val = st.session_state.aggrid_selected_cell_data['data'].get(col_name)
                if pd.notna(val) and str(val).strip() != "":
                    value_to_highlight_list.append({"column": col_name, "value": str(val)})
                selected_display_name = f"'{col_name}'"


            # Proceed with highlighting if there are values to highlight
            if value_to_highlight_list:
                selected_filename = selected_row_data.get("Filename")

                if not selected_filename:
                    st.warning("Selected row does not contain a 'Filename' column. Cannot link to PDF.")
                    st.session_state.current_display_image = None
                    st.session_state.current_display_caption = "Selected row has no 'Filename'."
                elif selected_filename not in st.session_state.pdf_ocr_data:
                    st.warning(f"OCR data for '{selected_filename}' not found. Please upload and process the PDF first.")
                    st.session_state.current_display_image = None
                    st.session_state.current_display_caption = "OCR data not found for selected file."
                else:
                    ocr_results_for_file = st.session_state.pdf_ocr_data[selected_filename]

                    # Get the initial image for the relevant page
                    initial_image = None
                    target_page_num = None
                    # Find the first valid page for any of the values to establish a base page for highlighting
                    for item_to_highlight in value_to_highlight_list:
                        temp_page_num, _, _ = find_best_ocr_match(item_to_highlight['value'], ocr_results_for_file)
                        if temp_page_num:
                            target_page_num = temp_page_num
                            break # Found a page, use this as the primary page for highlights
                    
                    if target_page_num:
                        for page_data in ocr_results_for_file:
                            if page_data["page_num"] == target_page_num:
                                initial_image = page_data["image"].copy()
                                break
                    
                    if initial_image:
                        highlighted_image = initial_image
                        all_match_scores = []
                        highlighted_fields = []

                        for item_to_highlight in value_to_highlight_list:
                            best_page_num_current, best_bbox_current, best_score_current = find_best_ocr_match(item_to_highlight['value'], ocr_results_for_file)
                            
                            # Only highlight if it's on the same target page and a good match
                            if best_bbox_current and best_page_num_current == target_page_num and best_score_current > 50: # Threshold for highlighting
                                highlighted_image = apply_highlight_to_image(highlighted_image, best_bbox_current, highlight_color=st.session_state.highlight_color)
                                all_match_scores.append(f"{item_to_highlight['column']}: {best_score_current:.2f}%")
                                highlighted_fields.append(f"'{item_to_highlight['column']}' ('{item_to_highlight['value']}')")

                        if highlighted_fields:
                            st.session_state.current_display_image = highlighted_image
                            caption_text = f"Highlighted on {selected_filename} - Page {target_page_num}:<br>"
                            caption_text += "<br>".join(highlighted_fields)
                            caption_text += f"<br>Match Scores: {', '.join(all_match_scores)}"
                            st.session_state.current_display_caption = caption_text
                        else:
                            st.session_state.current_display_image = None
                            st.session_state.current_display_caption = f"No strong OCR matches found for {selected_display_name} in '{selected_filename}'."

                    else:
                        st.session_state.current_display_image = None
                        st.session_state.current_display_caption = f"Image for relevant page not found for '{selected_filename}'."

            else: # No value to highlight based on selection (e.g., empty cell or 'All' with no valid fields)
                st.session_state.current_display_image = None
                st.session_state.current_display_caption = "No valid value selected for highlighting."

        else: # No row selected from AgGrid
            st.session_state.current_display_image = None
            st.session_state.current_display_caption = "Click a cell in the table to see its highlight on the PDF."
            # Clear column dropdown selection if no row is active
            if 'column_selector_dropdown' in st.session_state:
                st.session_state.column_selector_dropdown = '' # Reset to empty string
            if 'last_column_selection' in st.session_state:
                st.session_state.last_column_selection = ''


    else:
        st.info("Please upload your CSV data to see the extracted information.")

    st.header("5. PDF Visualizer with Highlights")
    if st.session_state.current_display_image:
        # Use st.markdown for caption to enable <br>
        st.image(st.session_state.current_display_image, use_column_width=True)
        st.markdown(st.session_state.current_display_caption, unsafe_allow_html=True)
    else:
        st.info(st.session_state.current_display_caption)


if __name__ == "__main__":
    main()