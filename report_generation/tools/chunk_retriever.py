#!/usr/bin/python3

from typing import List, Union, Type, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForToolRun

def load_chunk_retriever(vectordb):
  class ChunkRetrieverInput(BaseModel):
    query: str = Field(description = 'question that you want to look up')
  class ChunkRetrieverOutput(BaseModel):
    chunks: List[Document]
  class ChunkRetrieverConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    retriever: VectorStoreRetriever
  class ChunkRetrieverTool(StructuredTool):
    name: str = "Retrieves a small set of relevant document chunks from the corpus."
    description: str = "ONLY use for research questions that want to look up specific facts from the knowledge corpus, and don't need entire documents."
    args_schema: Type[BaseModel] = ChunkRetrieverInput
    config: ChunkRetrieverConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ChunkRetrieverOutput:
      docs = self.config.retriever.invoke(query)
      return ChunkRetrieverOutput(chunks = docs)
  retriever = vectordb.as_retriever(search_kwargs = {'k': 5})
  return ChunkRetrieverTool(config = ChunkRetrieverConfig(retriever = retriever))

