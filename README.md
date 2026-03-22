Azerbaijan AI Strategy RAG Bot
Bu layihə, Azərbaycan Respublikasının Süni İntellekt Strategiyası rəsmi sənədi əsasında hazırlanmış, yüksək dəqiqlikli RAG (Retrieval-Augmented Generation) sistemidir. Bot, sənəd daxilindəki strateji hədəfləri, qanunvericilik bəndlərini və fəaliyyət istiqamətlərini semantik olaraq analiz edərək istifadəçiyə rəsmi kontekstə uyğun cavablar təqdim edir.

🚀 Texniki Üstünlüklər
Custom Chunking Logic: Mətnin təbii strukturunu qorumaq üçün standart splitterlərdən fərqli olaraq, abzas və bölmə sonlarını nəzərə alan, ardıcıl məna ötürücülü (overlap) parçalama alqoritmi.

Two-Stage Retrieval (Re-ranking): 1.  Stage 1: FAISS və paraphrase-multilingual-MiniLM-L12-v2 ilə sürətli vektor axtarışı.
2.  Stage 2: cross-encoder/ms-marco-MiniLM-L-6-v2 ilə tapılmış hissələrin suala uyğunluq dərəcəsinə görə yenidən sıralanması.

Prompt Engineering: Hallüsinasiyaların qarşısını alan və modelin yalnız təqdim olunan sənəd daxilində cavab verməsini təmin edən "Strict Context Control" təlimatları.

Multilingual Support: Azərbaycan dilinin qrammatik və semantik özəlliklərini dəstəkləyən çoxdilli embedding modellərinin inteqrasiyası.

🛠 Texnologiya Steki
Dil: Python 3.10+

LLM: OpenAI GPT-4o / GPT-4o-mini

Orchestration: LangChain

Vector Store: FAISS (Facebook AI Similarity Search)

Embeddings: HuggingFace Transformers

Reranker: MS-MARCO Cross-Encoder

📈 Layihə Memarlığı
Sistem aşağıdakı ardıcıllıqla işləyir:

Ingestion: .txt formatlı strategiya sənədi təmizlənir və xüsusi overlap məntiqi ilə hissələrə bölünür.

Retrieval: İstifadəçinin sualı vektora çevrilir və FAISS bazasından ən yaxın 10 namizəd (candidate) seçilir.

Reranking: Seçilmiş 10 hissə Cross-Encoder modelinə ötürülür və ən yüksək skora malik 5 hissə LLM üçün kontekst kimi ayrılır.

Generation: GPT-4o-mini seçilmiş kontekst əsasında rəsmi üslubda yekun cavabı formalaşdırır.

📦 Quraşdırılma
1.Repozitoriyanı klonlayın:

git clone https://github.com/samadsafarov/Azerbaijan-AI-Strategy-RAG-Bot-.git
cd ai-strategy-rag-bot

2.Kitabxanaları yükləyin:


pip install -r requirements.txt

3.Konfiqurasiya:
Layihə qovluğunda .env faylı yaradın:


OPENAI_API_KEY=your_api_key_here

4.İşə salın:


python main.py