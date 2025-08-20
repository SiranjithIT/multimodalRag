from agent import MultiModalRagAgent
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
agent = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    query: str 

@app.post("/upload")
async def upload(pdf_file: UploadFile):
    global agent 
    agent = MultiModalRagAgent(pdf_file=pdf_file)
    return {"Response": f"File {pdf_file.filename} is processed and stored to vector store"}
    
@app.post("/query/")
async def llm_rag(userQuery: RequestBody):
    global agent
    if agent != None:
        answer = agent.multimodal_pdf_rag_pipeline(userQuery.query)
        return {"Response": answer}


if __name__ == "__main__":
  agent = MultiModalRagAgent("multimodal_sample.pdf")

  queries = [
      "What does the chart on page 1 show about revenue trends?",
      "Summarize the main findings from the document",
      "What visual elements are present in the document?"
  ]

  for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)
    answer = agent.multimodal_pdf_rag_pipeline(query)
    print(f"Answer: {answer}")
    print("=" * 70)