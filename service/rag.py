#contexual retrival with prompt chaning feedback and reduce latency
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from huggingface_hub import InferenceClient
import os
import hashlib
import time

# Load environment variables
load_dotenv()
# llm = ChatOpenAI(model= "typhoon2-8b-instruct",api_key="dummy", base_url="http://localhost:5555/v1")
llm = ChatOpenAI(model="typhoon-v2-70b-instruct", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.opentyphoon.ai/v1")  
# markdown = ocr_document('data', base_url='https://api.opentyphoon.ai/v1', api_key=os.getenv("OPENAI_API_KEY"))

#client = OpenAI(api_key=openai_key)
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN"),  # Hugging Face token for embedding
)
# === Define embedding function using Hugging Face ===
class E5EmbeddingFunction:
    def __init__(self, hf_client):
        self.client = hf_client

    def __call__(self, input):
        inputs = [f"passage: {text}" for text in input]
        return self.client.feature_extraction(
            inputs,
            model="intfloat/multilingual-e5-large"
        )

    def name(self):
        return "e5-huggingface-inference"

# Paths
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
print(f"Using data path: {DATA_PATH}")
# Load and process PDF documents
print("Loading documents from PDF files...")
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} documents from {DATA_PATH}")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
print("Splitting documents into chunks...")
chunks = splitter.split_documents(documents)

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = E5EmbeddingFunction(
    hf_client=hf_client
)
collection = chroma_client.get_or_create_collection(name="contextual_chunks", embedding_function=embedding_fn)

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""
CHUNK_CONTEXT_PROMPT = """
ตอบคำถามนี้โดยใช้เฉพาะบริบทด้านล่างนี้
<chunk>
{chunk_content}
</chunk>

กรุณาให้บริบทโดยย่อและกระชับ เพื่ออธิบายว่าส่วนนี้เกี่ยวข้องกับเอกสารฉบับเต็มอย่างไร
จุดประสงค์คือเพื่อช่วยให้การค้นหาเนื้อหา (search retrieval) มีประสิทธิภาพยิ่งขึ้น
โปรดตอบเฉพาะบริบทโดยย่อเท่านั้น และอย่าตอบข้อความอื่นใด
"""

def situate_context(doc: str, chunk: str) -> str:
    prompt = DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc) + "\n\n" + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
    response = llm.invoke(
        prompt, max_tokens=200, temperature=0.1
    )
    return response.content

full_doc_text = "\n\n".join([d.page_content for d in documents])
existing_ids = set(collection.get()["ids"])

for idx, doc_chunk in enumerate(chunks):
    chunk_text = doc_chunk.page_content
    uid = f"pdf_{idx}_" + hashlib.md5(chunk_text.encode("utf-8")).hexdigest()

    if uid in existing_ids:
        print(f"[!] Skipping chunk {idx} (already exists)")
        continue

    try:
        context = situate_context(chunk_text[:1500], chunk_text)
        full_chunk = context.strip() + "\n\n" + chunk_text
        collection.upsert(
            ids=[uid],
            documents=[full_chunk],
            metadatas=[{"chunk_index": idx, "context": context}]
        )
        print(f"[\u2713] Added chunk {idx}")
    except Exception as e:
        print(f"[\u2717] Failed to add chunk {idx}: {e}")

def ask_llm(messages) -> str:
    response = llm.invoke(messages, max_tokens=200, temperature=0.1)
    return response.content

def search_chunks(query, top_k=3):
    query_embedding =  "query: " + query
    results = collection.query(query_texts=[query_embedding], n_results=top_k)
    return results['documents'][0]

def generate_answer_with_feedback(query) -> str:
    chunks = search_chunks(query)
    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": """ท่านมีหน้าที่เลือกคำตอบที่ถูกต้องและเหมาะสมที่สุดจากบริบทที่ได้รับ โดยพิจารณาจากข้อมูลที่ปรากฏในบริบทดังกล่าว

ข้อกำหนดในการให้คำตอบ:

หากในบริบทมีการใช้คำที่เป็นการทับศัพท์ (transliteration) เช่น "ออร์โธปิดิกส์" ท่านสามารถใช้คำภาษาอังกฤษดั้งเดิม เช่น "Orthopedics" แทนได้

ให้เลือกคำตอบที่ถูกต้องเพียงข้อเดียวจากตัวเลือกที่ให้มา

การตอบกลับต้องอยู่ในรูปแบบของตัวเลือกตัวอักษรที่ถูกต้องที่สุด เช่น "ก" หรือ "ข" หรือ "ค" หรือ "ง" โดยต้องใส่เครื่องหมายคำพูดครอบตัวอักษรไว้เสมอ
คำตอบต้องอยู่ในรูปแบบ JSON ดังนี้:
{
  "answer": "ตัวเลือกที่ถูกต้อง เช่น ก",
  "reason": "คำอธิบายเหตุผล"
}
คำตอบให้แสดงในช่อง "answer" และต้องเป็นตัวเลือกที่ถูกต้องที่สุดจากตัวเลือกที่ให้มา
ให้แสดงคำอธิบาย ลงไปในช่อง "reason" เพื่ออธิบายเหตุผลที่เลือกคำตอบนั้น

ตัวอย่างรูปแบบที่ถูกต้อง: "ข" 
หากคำตอบไม่สามารถสรุปได้จากบริบทที่ให้มา ให้ตอบว่า "ไม่สามารถตอบได้จากข้อมูลที่มี" และไม่ต้องใส่คำอธิบาย"""
},
        {"role": "user", "content": f"""กรุณาตอบคำถามนี้โดยอิงจากเฉพาะบริบทที่ให้ไว้ด้านล่างเท่านั้น:

Context:
---------
{context}

Question:
{query}
"""}
    ]
    initial_answer = ask_llm(messages)

    feedback_prompt = [
    {"role": "system", "content": "คุณคือผู้ช่วยด้าน meta-reasoning"},
    {"role": "user", "content": f"""
คำถาม: "{query}"

คำตอบที่ได้รับ:
"{initial_answer}"

โปรดตรวจสอบว่า คำตอบนี้ได้รับการสนับสนุนอย่างเพียงพอจากบริบทที่ให้ไว้หรือไม่:
- ถ้า "ใช่" ให้ตอบเพียงว่า: คำตอบเพียงพอและได้รับการสนับสนุนอย่างดี
- ถ้า "ไม่ใช่" ให้เสนอ query ที่ควรใช้แทน โดยตอบในรูปแบบ:
query: [ข้อความค้นหาใหม่]
**ห้ามตอบคำอธิบายอื่น**"""
}
]
    feedback = ask_llm(feedback_prompt)

    if "suggest" in feedback.lower() or "improve" in feedback.lower():
        print(f"Feedback suggests refining query:\n{feedback}\n")
        refined_query = feedback.split("query")[-1].strip(":.\n\" ")
        new_chunks = search_chunks(refined_query)
        new_context = "\n\n".join(new_chunks)

        new_answer_prompt = [
            {"role": "system", "content": "คุณคือผู้ช่วยผู้เชี่ยวชาญที่ใช้เฉพาะเอกสารที่ให้ไว้เท่านั้น"},
            {"role": "user", "content": f"""ตอบคำถามนี้โดยใช้บริบทที่ปรับปรุงแล้ว:

Context:
---------
{new_context}

Original Question:
{query}
คำตอบต้องอยู่ในรูปแบบ JSON ดังนี้:
{
  "answer": "ตัวเลือกที่ถูกต้อง เช่น ก",
  "reason": "คำอธิบายเหตุผล"
}
**ห้ามใช้ข้อมูลนอกเหนือจากนี้ ห้ามเดาหรือคาดการณ์เอง**
**ให้ตอบเฉพาะตัวเลือกที่ถูกต้องในรูปแบบ: "ก" หรือ "ก,ข"**"""}
        ]
        final_answer = ask_llm(new_answer_prompt)
        return final_answer
    else:
        print(" Initial answer was sufficient.")
        return initial_answer

if __name__ == "__main__":
    from pydantic import BaseModel, Field
    class AnswerOutput(BaseModel):
        answer: str = Field(description="The answer to the question.")
        reason: str = Field(description="The reason for the answer.")

    while True:
                user_input = input("You: ")
                if user_input.lower() == "exit" or user_input.lower() == "quit":
                    break
                response = generate_answer_with_feedback(user_input)
                try:
                    parsed = AnswerOutput.model_validate_json(response)
                except Exception as e:
                    print(f"Error parsing response: {e}")
                    parsed = {"answer": "Invalid response format", "reason": str(e)}
                print(f"Answer: {parsed.answer}\nReason: {parsed.reason}")
