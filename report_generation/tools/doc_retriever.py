#!/usr/bin/python3

from typing import List, Union, Type, Optional
from pydantic import BaseModel, Field
from langchain_neo4j import Neo4jVector
from langchain.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForToolRun

def load_document_retriever(vectordb):
  class DocumentRetrieverInput(BaseModel):
    query: str = Field(description = 'question that you want to look up')
  class DocumentRetrieverOutput(BaseModel):
    chunks: List[Document]
  class DocumentRetrieverConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    retriever: VectorStoreRetriever
    vectordb: Neo4jVector
  class DocumentRetrieverTool(StructuredTool):
    name: str = 'Document retriever that retrieves entire documents from the corpus.'
    description: str = 'ONLY use for research questions that may require searching over entire research reports. Will be slower and more expensive than chunk-level retrieval but may be necessary.'
    args_schema: Type[BaseModel] = DocumentRetrieverInput
    config: DocumentRetrieverConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> DocumentRetrieverOutput:
      docs = self.config.retriever.invoke(query)
      patent_paths = set([doc.metadata['patent_path'] for doc in docs])
      chunks = list()
      for patent_path in patent_paths:
        matches = self.config.vectordb.query("match (a:Chunk {patent_path: '%s'}) return a;" % patent_path)
        for match in matches:
          a = match['a']
          doc = Document(page_content = a['text'])
          doc.metadata['page_num'] = a['page_num']
          doc.metadata['patent_path'] = a['patent_path']
          chunks.append(doc)
      return DocumentRetrieverOutput(chunks = chunks)
  retriever = vectordb.as_retriever(search_kwargs = {'k': 5})
  return DocumentRetrieverTool(config = DocumentRetrieverConfig(retriever = retriever, vectordb = vectordb))

