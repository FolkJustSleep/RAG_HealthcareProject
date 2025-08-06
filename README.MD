# RAG_HealthcareProject

As Thailand transitions into an aging society, the demand on medical personnel continues to grow, raising concerns about long-term healthcare capacity. 

To help address this challenge, we developed an AI agent system designed to assist medical professionals by providing contextual, accurate responses using Retrieval-Augmented Generation (RAG) technology.
  
Our AI agent can be applied in various healthcare-related scenarios, such as answering patient queries, assisting with medical documentation, and supporting decision-making—all based on domain-specific data.  

This system uses RAG (Retrieval-Augmented Generation) to ground AI responses in a custom knowledge base, allowing for flexible, updatable, and context-aware answers. By enriching the AI with more relevant data, its performance and accuracy improve over time. 

<h2> GET STARTED </h2>

Create Virture environtment

```
python3 -m venv .venv
```

Activate Environtment

```
source venv/bin/activate
```

Install Dependencies

```
pip install -r requirments.txt
```

if this have issue you can install 

```
pip install chromadb
pip install langchain-community
pip install openai
pip install pypdf
pip install python-dotenv
pip install pydantic
```

Run server

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server will run on port 8000 and you can use RAG API with this request

```
curl -X POST http://localhost:5000/rag \
    -H "Content-Type: application/json" \
    -d '{"question": "your-question"}'
```

or using Postman you can sent JSON like this

```
{
    "question" : "your-quesion"
}
```
