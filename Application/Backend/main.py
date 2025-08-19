from agent import MultiModalRagAgent
from fastapi import FastAPI


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