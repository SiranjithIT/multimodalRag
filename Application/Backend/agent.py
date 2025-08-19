import fitz
from langchain_core.documents import Document
from PIL import Image
import torch
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from model import llm, clip_model, clip_processor


class UploadProcess:
  def __init__(self, pdf_file):
    self.doc = fitz.open(pdf_file)
    self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    self.all_docs, self.embeddings_array = self._process(self.doc)
    self.vector_store = FAISS.from_embeddings(
      text_embeddings=[(doc.page_content, emb) for doc, emb in zip(self.all_docs, self.embeddings_array)],
      embedding=None,
      metadatas=[doc.metadata for doc in self.all_docs],
    )
  
  def embed_image(self, image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):
      image = Image.open(image_data).convert("RGB")
    else:
      image = image_data
    
    inputs=clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
      features = clip_model.get_image_features(**inputs)
      features = features / features.norm(dim=-1, keepdim=True)
      return features.squeeze().numpy()
  
  def embed_text(self, text):
    """Embed text using CLIP."""
    inputs = clip_processor(
      text=text, 
      return_tensors="pt", 
      padding=True,
      truncation=True,
      max_length=77
    )
    with torch.no_grad():
      features = clip_model.get_text_features(**inputs)
      features = features / features.norm(dim=-1, keepdim=True)
      return features.squeeze().numpy()
  
  def _process(self, doc):
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    for i, page in enumerate(doc):
      text = page.get_text()
      if text.strip():
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        text_chunks = self.splitter.split_documents([temp_doc])
        
        for chunk in text_chunks:
          embedding = self.embed_text(chunk.page_content)
          all_embeddings.append(embedding)
          all_docs.append(chunk)
      
      for img_index, img in enumerate(page.get_images(full=True)):
        try:
          xref = img[0]
          base_image = doc.extract_image(xref)
          image_bytes = base_image["image"]
          
          # Convert to PIL Image
          pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
          
          # Create unique identifier
          image_id = f"page_{i}_img_{img_index}"
          
          # Store image as base64 for later use with GPT-4V
          buffered = io.BytesIO()
          pil_image.save(buffered, format="PNG")
          img_base64 = base64.b64encode(buffered.getvalue()).decode()
          self.image_data_store[image_id] = img_base64
          
          # Embed image using CLIP
          embedding = self.embed_image(pil_image)
          all_embeddings.append(embedding)
          
          # Create document for image
          image_doc = Document(
            page_content=f"[Image: {image_id}]",
            metadata={"page": i, "type": "image", "image_id": image_id}
          )
          all_docs.append(image_doc)
          
        except Exception as e:
          print(f"Error processing image {img_index} on page {i}: {e}")
          continue  
    return all_docs, np.array(all_embeddings)
    
  def get_vector_store(self):
    return self.vector_store
          

class MultiModalRagAgent:
  def __init__(self, pdf_file):
    self.process = UploadProcess(pdf_file)
    self.vector_store = self.process.get_vector_store()
  
  def retrieve_multimodal(self, query, k=5):
    """Unified retrieval using CLIP embeddings for both text and images."""
    query_embedding = self.process.embed_text(query)
    
    results = self.vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    return results
  
  def create_multimodal_message(self, query, retrieved_docs):
    """Create a message with both text and images for GPT-4V."""
    content = []
    
    # Add the query
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in self.image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self.image_data_store[image_id]}"
                }
            })
    
    # Add instruction
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)
  
  def multimodal_pdf_rag_pipeline(self, query):
    """Main pipeline for multimodal RAG."""
    # Retrieve relevant documents
    context_docs = self.retrieve_multimodal(query, k=5)
    
    # Create multimodal message
    message = self.create_multimodal_message(query, context_docs)
    print(message)
    
    # Get response from GPT-4V
    response = llm.invoke([message])
    
    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")
    
    return response.content