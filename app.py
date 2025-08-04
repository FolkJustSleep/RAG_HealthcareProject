from fastapi import FastAPI
from model.requestmodel import Input
from gateway.rag_gateway import askllm

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Agents    API"}

@app.post("/rag")
def ask_rag(input: Input):
    parsed = askllm(input)
    return {"answer": parsed.answer, "reason": parsed.reason}
