
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. This router is designed for heavy duty usage and can handle high trafffic load. ",
        metadata={"type": "product"}
    ),
    Document(
        page_content="For network connectivity issues,first check the ethernet cable and router status lights.",
        metadata = {"type":"troubleshooting"}
    ),
    Document(
        page_content="To reset your monthly subscription, navigate to the 'billing' section in your account settings.",
        metadata = {"type":"billing"}
    ),
    Document(
        page_content="Error code E_CONN_REFUSED typically means the server is not accepting connections. Please check the firewall settings.",
        metadata = {"type":"error"}
    ),
    Document(
        page_content="To configure Single Sign-On (SSO) or multi-factor authentication, visit the security dashboard.",
        metadata = {"type":"auth"}
    ),
    Document(
        page_content="System configuration options can be modified in the settings panel under the admin preferences.",
        metadata = {"type":"config"}
    ),
    Document(
        page_content="All data processing practices are audited annually to ensure strict compliance with SOC 2 and GDPR regulations.",
        metadata = {"type":"compliance"}
    ),
    Document(
        page_content="To improve application performance and reduce latency, consider enabling the distributed caching layer in the configuration.",
        metadata={"type": "performance"}
    ),
    Document(
        page_content="Production deployments are managed via CI/CD pipelines, utilizing Docker containers and Kubernetes for orchestration.",
        metadata={"type": "deployment"}
    ),
    Document(
        page_content="API usage is strictly rate-limited to 1,000 requests per minute per IP address to ensure service stability.",
        metadata={"type": "api_rate_limit"}
    ),
    Document(
        page_content="Automated backups of all databases are taken daily and stored in a secondary geographic region for disaster recovery.",
        metadata={"type": "backup"}
    ),
    Document(
        page_content="Administrators can invite new team members and manage role-based access control (RBAC) via the user management console.",
        metadata={"type": "user_management"}
    )
]


print(f"Loaded {len(documents)} documents...")

vectorstore = Chroma.from_documents(
    documents,
    embeddings_model,
    collection_name = "hybrid_test"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k":3})

bm25_retriever = BM25Retriever.from_documents(documents, k=3)

ensemble = EnsembleRetriever(
    retrievers = [bm25_retriever,vector_retriever],
    weights = [0.5,0.5]
)


def test_query(query,name,retriever):
    results = retriever.invoke(query)
    print(f"\n {name} - Query: {query}")
    for i,doc in enumerate(results[:3]):
        preview = doc.page_content[:80]
        print(f"Result {i+1}: {preview}...")
        print(f"Metadata: {doc.metadata}")

queries = [
    "SKU-7742X specifications",
    "E_CONN_REFUSED error",
    "How do I reset my subscription"
    "router configuration"
]

for q in queries:
    test_query(q,"Vector DB",vector_retriever) 
    test_query(q,"BM25",bm25_retriever)
    test_query(q,"Hybrid Search",ensemble)
    print("\n" + "="*50 + "\n")