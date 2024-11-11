import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from sentence_transformers import SentenceTransformer
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.parsers import ParseUnstructured

# Pathway configuration
model = SentenceTransformer('all-MiniLM-L6-v2')
splitter = TokenCountSplitter(min_tokens=1, max_tokens=100)
embedder = SentenceTransformerEmbedder(model='all-MiniLM-L6-v2')
parser = ParseUnstructured()

# Initialize the Pathway Vector Store Server
data_sources = pw.io.fs.read(
    "./data",
    format="binary",
    mode="streaming",
    with_metadata=True,
)

# Initializing a VectorStoreServer object
vector_store_server = VectorStoreServer(
    data_sources,
    embedder=embedder,
    parser=parser,
    splitter=splitter,
)

# Configuring the host and port
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8756

def run_server():
    vector_store_server.run_server(
        host=PATHWAY_HOST,
        port=PATHWAY_PORT,
        with_cache=False,
        threaded=True,
    )

# Running the server
if __name__ == "__main__":
    run_server()
