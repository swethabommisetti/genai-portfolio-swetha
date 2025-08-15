import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict
from utils.supabase_utils import get_groq_api_key
#from src.tools.receipt_extraction import ProcessReceiptTool
#from src.tools.receipt_extraction import ProcessReceiptTool

from tools.receipt_extraction import ProcessReceiptTool
groq_key = get_groq_api_key()  # raises with a clear message if missing
llm = ChatGroq(
    api_key=groq_key,               # <-- pass explicitly, don’t rely only on env
    model="llama-3.1-8b-instant",
    temperature=0.0,
)

load_dotenv()
# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    )

# --- LangGraph State ---
class ReceiptState(TypedDict):
    image_path: str
    receipt_data: dict
    messages: list

# Instantiate the tool
receipt_tool = ProcessReceiptTool()

# --- Graph Nodes ---
def extract_receipt(state: ReceiptState) -> ReceiptState:
    """Step 1: Process the receipt image."""
    data = receipt_tool.run(state["image_path"])
    return {"image_path": state["image_path"], "receipt_data": data}


GENERATE_PROMPT = (
    "You are a summary generator."
    "Use the following pieces of context extracted from a store receipt of items bought to generate the answer. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Context: {context}"
)

def generate_final_answer(state: ReceiptState):
    """Generate an answer."""
    receipt_data = state["receipt_data"]
    prompt = GENERATE_PROMPT.format(context=receipt_data)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# --- Build Graph ---
workflow = StateGraph(ReceiptState)

workflow.add_node("extract_receipt", extract_receipt)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.set_entry_point("extract_receipt")
workflow.add_edge("extract_receipt", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

# --- Compile Graph ---
app = workflow.compile()

# --- Run Flow ---
import os, tempfile
from typing import Optional, Dict, Any
from pathlib import Path

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
    Wrapper used by the Streamlit UI. Saves bytes to a temp file (if provided)
    and then calls the existing LangGraph `app` you built:
        app.invoke({"image_path": <path>})
    Returns a simple dict payload for the UI to display.
    """
    # Prefer local bytes (more reliable than signed URL expiry)
    temp_path = None
    try:
        if file_bytes:
            # write to a temp file so your tool can read from path
            suffix = ""
            if filename and "." in filename:
                suffix = "." + filename.rsplit(".", 1)[-1]
            fd, temp_path = tempfile.mkstemp(prefix="receipt_", suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(file_bytes)
            image_path = temp_path
        elif file_url:
            # if your tool can accept URLs directly, you could pass file_url instead.
            # As your tool expects a path, download the URL here if needed.
            # For now, just pass the URL string – update if your tool requires a path.
            image_path = file_url
        else:
            raise ValueError("No file_bytes or file_url provided.")

        # Call your compiled graph
        result = app.invoke({"image_path": image_path})

        # Normalize response for UI
        out = {
            "filename": filename,
            "mime_type": mime_type,
            "user_email": user_email,
            "visitor_id": visitor_id,
            "bucket": bucket,
            "graph_result_keys": list(result.keys()) if isinstance(result, dict) else None,
            "result": result,
        }
        return out

    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


async def main():
    while True:
        query = input("Please enter your query:  ")
        if query == "q":
            break
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        async for s in app.astream(
        {"image_path": query},
        config=config):
            for node, update in s.items():
                print("Update from node:", node)
                if update.get("messages"):
                    print("AI Response : ", update["messages"][-1].content)
                    print("-----"*20)

                
                
if __name__ == "__main__":
    import asyncio
    
    loop= asyncio.get_event_loop()
    loop.run_until_complete(main())