from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize FastAPI
app = FastAPI()

# Define Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load Sentence Transformer for Embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load GPT-2 Model Globally (for faster response)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ✅ Initialize FAISS Index
faiss_index = None
stored_sentences = []

# ✅ Extract Text from PDF
def extract_text_from_pdf(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

# ✅ Upload PDF
@app.post("/upload_pdf/{filename}")
async def upload_pdf(filename: str, file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        return {"message": f"Successfully uploaded {filename}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ Extract Text from PDF
@app.post("/extract_text/{filename}")
async def extract_text(filename: str):
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    text = extract_text_from_pdf(pdf_path)
    
    text_file_path = os.path.join(UPLOAD_FOLDER, filename.replace(".pdf", ".txt"))
    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    
    return {
        "message": f"Text extracted and saved to {text_file_path}",
        "extracted_text": text[:500]  # ✅ Show only first 500 chars for preview
    }

# ✅ Generate Embeddings with Normalization
@app.post("/get_embeddings/{filename}")
async def get_embeddings(filename: str):
    global faiss_index, stored_sentences

    text_file_path = os.path.join(UPLOAD_FOLDER, filename.replace(".pdf", ".txt"))
    if not os.path.exists(text_file_path):
        raise HTTPException(status_code=404, detail="Text file not found")
    
    with open(text_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    stored_sentences = text.split("\n")

    embeddings = sentence_model.encode(stored_sentences, convert_to_tensor=True).cpu().numpy()
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()  # ✅ Normalize embeddings

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return {"message": f"Successfully generated embeddings for {len(stored_sentences)} sentences"}

# ✅ GPT-2 Response Generation (With Filtering)
def generate_gpt2_response(query: str, context: str):
    try:
        # ✅ GPT-2 has a smaller context window, so trim input
        max_input_length = 800
        trimmed_context = context[:max_input_length]

        prompt = f"""
        Answer the following question based on the document context. If the document does not contain an answer, reply: "I can only answer questions based on the document. Please ask a relevant question."

        Context:
        {trimmed_context}

        Question: {query}
        Answer:
        """
        
        # ✅ Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # ✅ Generate output with better parameters
        output = model.generate(
            inputs, 
            max_length=150,  # ✅ Shorter responses to avoid hallucinations
            num_return_sequences=1, 
            temperature=0.7, 
            top_p=0.9, 
            no_repeat_ngram_size=2
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # ✅ Ensure answer does not repeat input prompt
        answer_start = response.find("Answer:")
        if answer_start != -1:
            response = response[answer_start + len("Answer:"):].strip()
        
        # ✅ If output is nonsense, return a default message
        if len(response) < 10 or "_____" in response:
            return "I can only answer questions based on the document. Please ask a relevant question."
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ✅ Search & Respond with GPT-2
@app.get("/search/{query}")
async def search_embeddings(query: str):
    if faiss_index is None or len(stored_sentences) == 0:
        raise HTTPException(status_code=500, detail="Embeddings not available. Generate them first.")

    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().numpy()
    query_embedding = F.normalize(torch.tensor(query_embedding), p=2, dim=1).numpy()

    D, I = faiss_index.search(query_embedding, k=5)  # ✅ Retrieve 5 sentences for better context

    # ✅ Filter out low-confidence results
    top_sentences = []
    for idx, distance in zip(I[0], D[0]):
        if idx < len(stored_sentences) and distance < 0.5:  # ✅ Only keep relevant results
            top_sentences.append(stored_sentences[idx])

    if not top_sentences:  
        return {
            "query": query,
            "response": "I can only answer questions based on the document. Please ask a relevant question."
        }

    context = " ".join(top_sentences[:3])  # ✅ Use only top 3 sentences for better relevance
    response = generate_gpt2_response(query, context)  # ✅ Call GPT-2 response function

    return {
        "query": query,
        "response": response
    }


