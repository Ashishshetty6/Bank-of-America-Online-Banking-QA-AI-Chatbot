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
# Load Embeddings (MUST MATCH your training!)
# ============================================
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


# ============================================
# Load FAISS Vector Store
# ============================================
print("🔍 Loading FAISS vectorstore...")

loaded_vectorstore = FAISS.load_local(
    "faiss_bank_terms_bge_m3",     # 🟢 THE CORRECT DIRECTORY
    embeddings=embeddings,
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
You are a helpful AI assistant who answers questions using ONLY the content from the Bank of America Online Banking Policy document.

Context:
{context}

Question:
{question}

Provide a clear and accurate answer using ONLY the context.
If the answer is not present in the context, respond:
"I could not find this information in the bank's policy document."
"""
)


# ============================================
# RAG Pipeline Function
# ============================================
def rag_pipeline(query):

    try:
        # 1. Retrieve relevant chunks
        docs = retriever.invoke(query)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 2. Build final prompt
        final_prompt = prompt.format(context=context_text, question=query)

        # 3. Call LLM
        response = llm.invoke(final_prompt)

        return response.content, context_text

    except Exception as e:
        return f"❌ ERROR: {str(e)}", ""


# ============================================
# Gradio UI
# ============================================
demo = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(
        label="💬 Ask a Question",
        placeholder="Example: What are the limits for Zelle payments?",
        lines=1
    ),
    outputs=[
        gr.Textbox(label="📘 AI Answer", lines=10),
        gr.Textbox(label="📄 Retrieved Context", lines=20)
    ],
    title="💼 Bank Terms & Services RAG Chatbot",
    description=(
        "This chatbot uses your PDF + FAISS + BGE-M3 embeddings + Groq LLM "
        "to answer questions strictly based on the Online Banking Service Agreement."
    ),
)

if __name__ == "__main__":
    demo.launch()
