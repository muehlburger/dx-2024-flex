import numpy as np
import uuid
import pandas as pd
from itertools import islice
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import OllamaEmbeddings
#embeddings = OllamaEmbeddings()
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# docker pull qdrant/qdrant
# docker run -p 6333:6333 qdrant/qdrant

COLLECTION_NAME = "powertrain"
NUM_OF_ROWS_TO_READ = 300000

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
)

# eventual list of books
MEASUREMENTS = ["./../data/powertrain/correctBehavior/TB3_correct_behavior_load_0_rounded_and_text_encoded_with_is_num2words.csv"]

# Client
client = QdrantClient(
        host="qdrant.ist.tugraz.at", 
        port=6333, 
        timeout=60*60,
)

def make_collection(client, collection_name: str):
    """
    Use 1st time on project
    :param client: qdrant client obj
    :type client: client
    :param collection_name: name of collection
    :type collection_name: str
    """

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )

make_collection(client,COLLECTION_NAME)

# Load Our Text File and split into chunks
def make_chunks(input_text: str):
    """
    Split text into chunks
    :param input_text: the source file
    :type input_text: str
    :return: chunks of text
    :rtype: qd1.texts
    """

    df = pd.read_csv(input_text, header=None, names=[ COLLECTION_NAME ], sep=",", nrows=NUM_OF_ROWS_TO_READ)
    print(f"Number of rows: {len(df)}")
    loader = DataFrameLoader(df, page_content_column=COLLECTION_NAME)
    data = loader.load()

    TEXT_SPLITTER_CHUNK_PARAMS = {
        "chunk_size": 2000,
        "chunk_overlap": 0,
        "length_function": len,
        "add_start_index": False,
        "separators": "\n\n"
    }

    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CHUNK_PARAMS)
    chunks = text_splitter.split_documents(data)

    return chunks


texts = make_chunks(MEASUREMENTS[0])

def gen_vectors(texts, embeddings):
    vectors = embeddings.embed_documents(
        [item.page_content for item in texts],
    )
    payload = [item for item in texts]
    payload = list(payload)
    #vectors = [v.tolist() for v in vectors]

    return vectors, payload

fin_vectors, fin_payload = gen_vectors(texts, embeddings)

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def upsert_to_qdrant(fin_vectors, fin_payload):
    """
    Add our vectors and meta into the Vector Database
    :param fin_vectors: _description_
    :type fin_vectors: generator
    :param fin_payload: _description_
    :type fin_payload: list
    """
    payloads = []
    for i in tqdm(range(len(fin_vectors))):
        payloads.append({"text": fin_payload[i].page_content })
    
    vectors = np.array(fin_vectors).tolist()

    for batch in tqdm(batched(zip(vectors, payloads), 100)):
        vectors, payloads = zip(*batch)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=[str(uuid.uuid4()) for _ in range(len(vectors))],
                vectors=vectors,
                payloads=payloads,
            )
        )
    # client.upsert(
    #     collection_name=COLLECTION_NAME,
    #     points=models.Batch(
    #         ids=[str(uuid.uuid4()) for _ in range(len(vectors))],
    #         vectors=vectors,
    #         payloads=payloads,
    #     )
    # )

    # client.update_collection(
    #    collection_name=COLLECTION_NAME,
    #     optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    # )

upsert_to_qdrant(fin_vectors, fin_payload)