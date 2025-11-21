import os
import gradio as gr
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================
# Load Environment Variables
# ============================================
load_dotenv()
os.environ["GROK_API_KEY"] = os.getenv("GROK_API_KEY")


# ============================================
# Load Embeddings (MUST MATCH training!)
# ============================================
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


# ============================================
# Load FAISS Vector Store
# ============================================
print("🔍 Loading FAISS vectorstore...")

loaded_vectorstore = FAISS.load_local(
    "faiss_index",  # Your saved directory
    embeddings=embeddings,  # REQUIRED for retriever
    allow_dangerous_deserialization=True
)

retriever = loaded_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ============================================
# Initialize Groq LLM
# ============================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ["GROK_API_KEY"]
)


# ============================================
# Prompt Template
# ============================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant who answers questions using ONLY the provided bank policy document.

Context:
{context}

Question:
{question}

Answer clearly and strictly based on the context.
If the answer is NOT found in the context, reply with:
"I could not find this information in the bank's policy document."
"""
)


# ============================================
# RAG Pipeline Function
# ============================================
def rag_pipeline(query):

    # 1. Retrieve relevant chunks
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Create final prompt
    final_prompt = prompt.format(context=context_text, question=query)

    # 3. LLM generates answer
    response = llm.invoke(final_prompt)

    return response.content, context_text


# ============================================
# Gradio UI
# ============================================
demo = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(
        label="Ask any question about Bank of America Online Banking Terms",
        placeholder="e.g. What does Bank of America warn about using Zelle?",
        lines=1
    ),
    outputs=[
        gr.Textbox(label="📘 AI Answer", lines=10),
        gr.Textbox(label="📄 Retrieved Contexts", lines=20)
    ],
    title="💼 Bank Terms & Services RAG Chatbot",
    description="Uses your PDF + FAISS + BGE-M3 Embeddings + Groq LLM for accurate answers based on the Bank of America Online Banking Agreement."
)

if __name__ == "__main__":
    demo.launch()
