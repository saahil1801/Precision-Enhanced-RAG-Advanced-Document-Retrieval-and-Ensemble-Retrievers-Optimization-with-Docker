from __future__ import annotations
import streamlit as st
import pandas as pd
import tempfile
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever

from typing import Dict, Optional, Sequence
from langchain.schema import Document
from langchain.pydantic_v1 import Extra, root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from sentence_transformers import CrossEncoder

def display_top_bge_ranked_docs(docs):
    st.sidebar.title("Top Ranked Documents")
    for i, doc in enumerate(docs):
        with st.sidebar.expander(f"Document {i+1} (Score: {doc.metadata['relevance_score']:.2f})", expanded=False):
            st.write(doc.page_content[:500])  # Display first 500 characters of the document
            st.write(f"**Full Content:** [Expand](javascript:void(0))")


def on_btn_click():
    del st.session_state.chat_history[:]

def main():

    # Set up the customization options
    with st.container():
        col1, col2= st.columns([.9, .15])
        with col1:  
            st.title('Advanced Groq RAG Assistant')
        with col2:
             st.button("Clear message", on_click=on_btn_click)
    
    st.sidebar.title('Customization')
    api_key_input = st.sidebar.text_input("Enter your GROQ API Key", type="password", label_visibility="collapsed")
    
    # if api_key_input.strip() == "":
    #     st.sidebar.warning("Please enter a valid GROQ API Key.")
    
    model = st.sidebar.selectbox(
        'Choose a model',
        ["mixtral-8x7b-32768", "llama3-8b-8192", "llama3-70b-8192"]
    )
    try:
        
        llm = ChatGroq(
                temperature=0, 
                groq_api_key=api_key_input, 
                model_name=model
            )
    except Exception as e:
    
         st.warning("An error occurred: {}".format('Please enter a valid GROQ API Key.'))
    
    
   

    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        
        
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                model_kwargs={'device': 'cpu'})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1250,
            chunk_overlap = 100,
            length_function = len,
            is_separator_regex = False
            )

        text_chunks = text_splitter.split_documents(text)

        vectorstore = Chroma(embedding_function=embeddings,
                            persist_directory="chromadb",
                            collection_name="full_documents")
            
        vectorstore.add_documents(text_chunks)
        vectorstore.persist()

        bm25_retriever = BM25Retriever.from_documents(text_chunks)
        bm25_retriever.k=10
        vs_retriever = vectorstore.as_retriever(search_kwargs={"k":10})
            #

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,vs_retriever],
                                                weight=[0.5,0.5])
            #

        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
            #
        reordering = LongContextReorder()
            #
        reranker = BgeRerank()

        
            #
        pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter,reordering,reranker])
            #
        compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                                base_retriever=ensemble_retriever)
        
           
        
        # docs = compression_pipeline.get_relevant_documents(query)
        # pretty_print_docs(docs)
        # query=st.text_input('Enter your query')
        
        
        if query := st.chat_input("What is up?") :

            docs = compression_pipeline.get_relevant_documents(query)
        
            ranked_docs = reranker.compress_documents(docs, query)

                # Display the ranked documents in the sidebar
            display_top_bge_ranked_docs(ranked_docs) 

            prompt_template = """
                Use the following pieces of information to answer the user's question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                Context: {context}
                Question: {question}

                Answer the question to the point

                Responses should be properly formatted to be easily read.

            """
            QA_PROMPT = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
            memory = ConversationBufferWindowMemory(k=10, input_key="question",memory_key='chat_history')
            
            qa_advanced = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=compression_pipeline,
                return_source_documents=False,
                memory=memory,
                combine_docs_chain_kwargs={'prompt': QA_PROMPT}
            )

            print("Current chat history:", st.session_state.chat_history)
            
            qa_adv_response = qa_advanced({"question": query, "chat_history": st.session_state.chat_history})
            
            st.session_state.chat_history.append((query, qa_adv_response['answer']))
            display_chat_history()
            

        


def display_chat_history():
    for i, conversation in enumerate(st.session_state.chat_history):
                                with st.chat_message("user"):
                                    st.markdown(f"**User**: {conversation[0]}")
                                with st.chat_message("assistant"):
                                    st.markdown(f"**Assistant**: {conversation[1]}")
                        
    
        

def pretty_print_docs(docs):
  print(
      f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n + {d.page_content}" for i,d in enumerate(docs)])
  )
    
class BgeRerank(BaseDocumentCompressor):
    model_name:str = 'BAAI/bge-reranker-large'
    """Model name to use for reranking."""
    top_n: int = 3
    """Number of documents to return."""
    model:CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""
    

    def bge_rerank(self,query,docs):
        model_inputs =  [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results
    
    


if __name__ == "__main__":
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    main()
