#!/usr/bin/python3

from typing import List, Union, Type, Optional
from pydantic import BaseModel, Field
from langchain_neo4j import Neo4jVector
from langchain.tools import StructuredTool
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.retrievers import MultiVectorRetriever
from langchain.storage._lc_store import create_kv_docstore

def load_document_retriever(vectordb, store):
  class DocumentRetrieverInput(BaseModel):
    query: str = Field(description = 'question that you want to look up')
  class DocumentRetrieverOutput(BaseModel):
    chunks: List[Document]
  class DocumentRetrieverConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    retriever: MultiVectorRetriever
  class DocumentRetrieverTool(StructuredTool):
    name: str = 'Document retriever that retrieves entire documents from the corpus.'
    description: str = 'ONLY use for research questions that may require searching over entire research reports. Will be slower and more expensive than chunk-level retrieval but may be necessary.'
    args_schema: Type[BaseModel] = DocumentRetrieverInput
    config: DocumentRetrieverConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> DocumentRetrieverOutput:
      docs = self.config.retriever.invoke(query)
      return DocumentRetrieverOutput(chunks = docs)
  child_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  retriever = ParentDocumentRetriever(
    vectorstore = vectordb,
    docstore = create_kv_docstore(store),
    child_splitter = child_splitter,
  )
  return DocumentRetrieverTool(config = DocumentRetrieverConfig(retriever = retriever))
