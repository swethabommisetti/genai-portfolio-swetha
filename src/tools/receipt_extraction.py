import os
import json
import base64

from mistralai import Mistral # Assuming you're using a Mistral LLM for the extraction step
from dotenv import load_dotenv
from langchain.tools import BaseTool
#from src.tools.receipt_schema import ReceiptData
from mistralai.extra import response_format_from_pydantic_model
from tools.receipt_schema import ReceiptData

# Load .env from the project root
load_dotenv()

# Load your Mistral API key (from environment variable or .env file)
#api_key = os.getenv("MISTRAL_API_KEY")
from utils.supabase_utils import get_mistral_api_key
api_key = get_mistral_api_key()     # raises with a helpful message if missing
os.environ.setdefault("MISTRAL_API_KEY", api_key)  # for SDKs that auto-read env

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")

# Path to your receipt image
receipt_image_path = "data/gaint_receipt1.png" # Replace with your image path

# Example: Encode your receipt image to Base64
def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None



def add_common_fields(data: dict) -> dict:
    """Add common fields to the receipt data."""
    for item in data.get("items", []):
        item.update({
            "store": data.get("store_name", ""),
            "address": data.get("address", ""),
        })
    return data

def process_receipt(encoded_image: str):
    try:
        # Create the structured OCR request using the "document_annotation" type
        # Mistral OCR, powered by the latest OCR model 'mistral-ocr-latest',
        # can extract text and structured content from PDF documents and images.
        # It supports multiple formats including image_url and document_url.
        client = Mistral(api_key=api_key)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest", # Use the latest OCR model
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_image}"
            },
            include_image_base64=True,
            document_annotation_format=response_format_from_pydantic_model(ReceiptData) 
        )

        # Access the structured output (which will be in JSON format)
        extracted_data = ocr_response.document_annotation
        extracted_json = json.loads(extracted_data)
        print("Extracted JSON data:")
        print(extracted_json)
        
        updated_data = add_common_fields(extracted_json)
        return updated_data
    except Exception as e:
        print(f"An error occurred during OCR or data extraction: {e}")
        return None


"""
class ProcessReceiptTool(BaseTool):
    name: str = "process_receipt"
    description: str = "Takes an image file path, processes a store receipt, extracts the data and returns structured output."

    def _run(self, image_path: str) -> dict:
        # Encode the image
        if not os.path.exists(image_path):
            return f"Error: The file {image_path} was not found. Please provide a valid existing path."
        encoded_image = encode_image_to_base64(image_path)

        result = process_receipt(encoded_image)
        return result

    async def _arun(self, image_path: str) -> dict:
        return self._run(image_path)
"""

import os
import tempfile
import mimetypes

try:
    import requests
except ImportError:
    requests = None  # we'll error nicely if a URL is used without requests


class ProcessReceiptTool(BaseTool):
    name: str = "process_receipt"
    description: str = (
        "Takes an image file path OR http(s) URL, processes a store receipt, "
        "extracts the data and returns structured output."
    )

    def _run(self, image_path: str) -> dict:
        """

        `image_path` can be a local filesystem path OR an http(s) URL.
        We download URLs to a temp file so we can reuse encode_image_to_base64()
        unchanged (ensures identical encoding format to your existing flow).
        """
        # --- URL branch ---
        if self._is_url(image_path):
            if requests is None:
                return ("Error: `requests` is required to fetch image URLs. "
                        "Install it with `pip install requests`.")
            tmp_path = None
            try:
                tmp_path = self._download_url_to_temp(image_path)
                encoded_image = encode_image_to_base64(tmp_path)
            except Exception as e:
                return f"Error: Failed to download/process URL: {e}"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            return process_receipt(encoded_image)

        # --- Local file branch (original behavior) ---
        if not os.path.exists(image_path):
            return f"Error: The file {image_path} was not found. Please provide a valid existing path."

        encoded_image = encode_image_to_base64(image_path)
        result = process_receipt(encoded_image)
        return result

    async def _arun(self, image_path: str) -> dict:
        return self._run(image_path)

    # ----------------- helpers -----------------
    def _is_url(self, s: str) -> bool:
        return isinstance(s, str) and s.lower().startswith(("http://", "https://"))

    def _download_url_to_temp(self, url: str) -> str:
        """
        Downloads the URL to a temp file and returns the file path.
        Tries to preserve extension based on Content-Type or URL.
        """
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content

        # guess extension
        ext_hint = None
        ctype = resp.headers.get("Content-Type")
        if ctype:
            ext_hint = mimetypes.guess_extension(ctype.split(";")[0].strip())
        if not ext_hint:
            ext_hint = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or "")
        if ext_hint in (".jpe",):
            ext_hint = ".jpg"

        fd, path = tempfile.mkstemp(prefix="receipt_", suffix=ext_hint or ".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        return path


