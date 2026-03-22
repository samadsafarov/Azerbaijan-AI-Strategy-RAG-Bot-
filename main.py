import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

def load_and_chunk(txt_path="ai_strategy.txt", chunk_size=1100, overlap_size=150):
    loader = TextLoader(txt_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\nBÖLMƏ ", "\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = text_splitter.split_documents(documents)

    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            prev_content = raw_chunks[i - 1].page_content
            prev_end = prev_content[-overlap_size:]

            first_space = prev_end.find(' ')
            if first_space != -1:
                prev_end = prev_end[first_space:].strip()

            new_content = f"... {prev_end}\n\n{chunk.page_content}"
            final_chunks.append(Document(
                page_content=new_content,
                metadata=chunk.metadata
            ))
    return final_chunks

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

INDEX_PATH = "faiss_ai"

if os.path.exists(INDEX_PATH):
    print("--- Baza diskdən yüklənir... ---")
    vector_db = FAISS.load_local(
        INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    print("--- Baza tapılmadı. Yeni məntiqlə qurulur... ---")
    chunks = load_and_chunk("ai_strategy.txt")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local(INDEX_PATH)

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_best_context(query, db):
    initial_docs = db.similarity_search(query, k=10)
    texts = [doc.page_content for doc in initial_docs]
    scores = reranker.predict([(query, txt) for txt in texts])
    best_indices = np.argsort(scores)[::-1][:5]
    return "\n\n".join([texts[i] for i in best_indices])

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1500
)

prompt_template = ChatPromptTemplate.from_template("""
Sən Azərbaycan Respublikasının Süni İntellekt Strategiyası üzrə rəsmi mütəxəssissən.
Sualı cavablandırmaq üçün YALNIZ aşağıdakı KONTEKST-dən istifadə et. 

TƏLİMATLAR:
- Sənəddəki siyahıları, bəndləri və maddələri (məs: 5.1.1) tam və ardıcıl şəkildə qeyd et.
- Rəsmi və aydın Azərbaycan dilində cavab ver.
- Cavab kontekstdə yoxdursa, uydurma, sadəcə sənəddə tapılmadığını bildir.

KONTEKST:
{context}

SUAL: {user_input}

CAVAB:""")

print("Strategiya üzrə Süni İntellekt Köməkçisi")
print("(Çıxmaq üçün 'stop', 'exit' və ya 'çıxış' yazın)\n")

while True:
    user_input = input("Sualınız: ")

    if user_input.lower() in ["stop", "exit", "çıxış"]:
        print("\nBot dayandırıldı. Sağ olun!")
        break

    if not user_input.strip():
        continue

    try:
        context = get_best_context(user_input, vector_db)
        formatted_prompt = prompt_template.format(context=context, user_input=user_input)
        response = llm.invoke(formatted_prompt)
        print(f"\nAI: {response.content}\n")
    except Exception as e:
        print(f"\nXəta baş verdi: {e}")

    print("-" * 30)