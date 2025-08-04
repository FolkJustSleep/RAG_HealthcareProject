# RAG_HealthcareProject

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

# Docker

if you have dependencies problem you can run docker-compose 

```
docker-compose up --build
```

To check, is docker are running
```
docker ps
```
