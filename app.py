import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# global chain (lazy loaded)
chain = None


def load_chain_once():
    global chain
    if chain is not None:
        return

    print("🔄 Loading model and vector store (one-time)...")

    from langchain import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.llms import CTransformers
    from src.helper import template

    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={"max_new_tokens": 128, "temperature": 0.01}
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    print("✅ Model loaded successfully")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "GET":
        return jsonify({"response": ""})

    load_chain_once()

    user_input = request.form.get("question", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a question."})

    result = chain({"query": user_input})
    return jsonify({"response": result["result"]})


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8080,
        debug=False,
        threaded=False,
        use_reloader=False
    )
