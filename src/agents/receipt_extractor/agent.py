# src/agents/receipt_extractor/agent.py

import uuid
import os
import tempfile
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from utils.supabase_utils import get_groq_api_key
from tools.receipt_extraction import ProcessReceiptTool

# ---- env hygiene --------------------------------------------------------------
load_dotenv()
# Silence the tracer warning by unsetting the var if present
os.environ.pop("LANGCHAIN_TRACING_V2", None)

# ---- tiny helper to mimic LangChain message objects (has .content) -----------
class SimpleMsg:
    def __init__(self, content: str):
        self.content = content

# ---- LangGraph state ----------------------------------------------------------
class ReceiptState(TypedDict):
    image_path: str
    receipt_data: dict
    messages: list

receipt_tool = ProcessReceiptTool()

def extract_receipt(state: ReceiptState) -> ReceiptState:
    """Step 1: Process the receipt image (your OCR/parse tool)."""
    data = receipt_tool.run(state["image_path"])
    return {"image_path": state["image_path"], "receipt_data": data}

GENERATE_PROMPT = (
    "You are a summary generator. "
    "Use the following pieces of context extracted from a store receipt of items bought to generate the answer. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Context: {context}"
)

def _summarize_with_langchain_groq(prompt: str) -> str:
    """Try LangChain-Groq first (will fail under some version combos)."""
    # Lazy import to avoid import-time deepcopy/pickle issues
    from langchain_groq import ChatGroq  # may raise

    groq_key = get_groq_api_key()
    llm = ChatGroq(
        api_key=groq_key,
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    # resp can be an AIMessage or similar; try to extract text
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)

def _summarize_with_groq_sdk(prompt: str) -> str:
    """Fallback: use Groq’s official SDK (avoids LangChain/Pydantic entirely)."""
    from groq import Groq  # pip install groq
    client = Groq(api_key=get_groq_api_key())
    out = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return out.choices[0].message.content

def generate_final_answer(state: ReceiptState) -> Dict[str, Any]:
    """Step 2: Summarize with LLM; LC-Groq first, otherwise Groq SDK fallback."""
    receipt_data = state.get("receipt_data", {})
    prompt = GENERATE_PROMPT.format(context=receipt_data)

    text = None
    try:
        text = _summarize_with_langchain_groq(prompt)
    except Exception:
        # Fallback path that avoids Pydantic/LangChain entirely
        text = _summarize_with_groq_sdk(prompt)

    return {"messages": [SimpleMsg(text or "")]}

# ---- Build graph --------------------------------------------------------------
workflow = StateGraph(ReceiptState)
workflow.add_node("extract_receipt", extract_receipt)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.set_entry_point("extract_receipt")
workflow.add_edge("extract_receipt", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

app = workflow.compile()

# ---- Public entrypoint for Streamlit -----------------------------------------
def extract_receipt_values(
    file_url: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    user_email: Optional[str] = None,
    visitor_id: Optional[str] = None,
    bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Saves bytes to a temp file (if provided) and runs the graph:
        app.invoke({"image_path": <path or URL>})
    """
    temp_path = None
    try:
        if file_bytes:
            suffix = ""
            if filename and "." in filename:
                suffix = "." + filename.rsplit(".", 1)[-1]
            fd, temp_path = tempfile.mkstemp(prefix="receipt_", suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(file_bytes)
            image_path = temp_path
        elif file_url:
            image_path = file_url
        else:
            raise ValueError("No file_bytes or file_url provided.")

        result = app.invoke({"image_path": image_path})

        return {
            "filename": filename,
            "mime_type": mime_type,
            "user_email": user_email,
            "visitor_id": visitor_id,
            "bucket": bucket,
            "graph_result_keys": list(result.keys()) if isinstance(result, dict) else None,
            "result": result,
        }
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

# Optional local CLI
if __name__ == "__main__":
    import asyncio
    async def main():
        while True:
            query = input("Enter an image path or URL (q to quit): ")
            if query.lower().strip() == "q":
                break
            res = app.invoke({"image_path": query})
            msgs = res.get("messages") if isinstance(res, dict) else None
            if msgs:
                print("AI Response:", getattr(msgs[-1], "content", msgs[-1]))
            print("-----" * 20)
    asyncio.run(main())
