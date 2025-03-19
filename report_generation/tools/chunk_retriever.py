#!/usr/bin/python3

from typing import List, Union, Type, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_neo4j import Neo4jVector
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.retrievers import MultiVectorRetriever
from langchain.storage._lc_store import create_kv_docstore

def load_chunk_retriever(vectordb, store):
  class ChunkRetrieverInput(BaseModel):
    query: str = Field(description = 'question that you want to look up')
  class ChunkRetrieverOutput(BaseModel):
    chunks: List[Document]
  class ChunkRetrieverConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    retriever: MultiVectorRetriever
  class ChunkRetrieverTool(StructuredTool):
    name: str = "Retrieves a small set of relevant document chunks from the corpus."
    description: str = "ONLY use for research questions that want to look up specific facts from the knowledge corpus, and don't need entire documents."
    args_schema: Type[BaseModel] = ChunkRetrieverInput
    config: ChunkRetrieverConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ChunkRetrieverOutput:
      docs = self.config.retriever.invoke(query)
      return ChunkRetrieverOutput(chunks = docs)
  child_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
  retriever = ParentDocumentRetriever(
    vectorstore = vectordb,
    docstore = create_kv_docstore(store),
    child_splitter = child_splitter,
    parent_splitter = parent_splitter
  )
  return ChunkRetrieverTool(config = ChunkRetrieverConfig(retriever = retriever))
