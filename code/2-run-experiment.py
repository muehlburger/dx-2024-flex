from ast import arg
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from langchain.chains.question_answering import load_qa_chain
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--start", default=3990, type=int, help="Start index of time series to use")
parser.add_argument("-e", "--end", default=4110, type=int, help="Stop index of time series to use")
parser.add_argument("-m", "--model", default="mistral", help="LLM model to use (mistral, mixtral, gemma, codegemma, command-r, codellama, qwen:14b, mistral-openorca, dolphin-mixtral, llama2-uncensored, llama3, llama3.1:70b)")
parser.add_argument("-d", "--device", default="cuda:0", help="Device to use for LLM")
parser.add_argument("-ss", "--search_strategy", default="similarity", help="Search type of vectore store (possible values 'similarity', 'mmr', 'random')")
parser.add_argument("-n", "--num_of_results", default=10, type=int, help="Number of documents to retrieve from vector store")
parser.add_argument("-c", "--collection", default="measurements", help="Name of the collection in Qdrant")
args = vars(parser.parse_args())
 
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

logger=logging.getLogger()
logger.setLevel(logging.INFO)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs = {'device': args['device']})

COLLECTION_NAME = args["collection"]
START = args["start"]
END = args["end"]

fault1 = "./../data/battery_runtime_fault1_text_encoded_with_is_num2words.csv"
fault1_rounded = "./../data/battery_runtime_fault1_rounded_and_text_encoded_with_is_num2words.csv"
powertrain_f1 = "./../data/powertrain/faultyBehavior/TB3_f1_behavior_load_0_rounded_and_text_encoded_with_is_num2words.csv"

client = QdrantClient(
        host="qdrant.ist.tugraz.at",
        port=6333, 
        timeout=60,
)

# client.update_collection(
#     collection_name=COLLECTION_NAME,
#     optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
# )

qdrant = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
    metadata_payload_key="metadata",
    content_payload_key="text",
)

df = pd.read_csv(powertrain_f1, header=None, names=[COLLECTION_NAME])
loader = DataFrameLoader(df, page_content_column=COLLECTION_NAME)
data = loader.load()

model = args["model"]

llm = ChatOllama(model=model,
                 temperature=0,
                 device=args["device"],
)

template = """
Use the following context to answer the question at the end. If you cannot determine, just say no, don't try to make up an answer. Use short, technical answers like (zero anomalies, yes, no). Write a response that appropriately completes the request.

Context: Normal operating range:
{context}

Question: {question}

Test Measurements:
{measurement}

Provide an anomaly rating between 0-100 based on the following criteria:

1) Values are in a reasonable range of the provided context.
2) Values do not contain anomalies.
3) Values are consistent with the given context.

Check *very carefully* with the given ranges before giving your rating.

Answer:
- Anomaly: e.g. yes, no
- Anomaly-Rating:
- Explanation:

"""

template2 = """
Analyze the provided test measurements to detect potential faults in all areas of the powertrain system using the provided context and fault models. You must evaluate the detected fault model for each component (Battery, H-Bridge, Motor, Load) and provide an assessment for each. If a fault is detected, identify the specific fault type and the affected area (e.g., Motor: Short, Battery: Empty, H-Bridge: F2, Load: OK). If you cannot conclusively determine the fault, respond with "uncertain." Use short, technical answers and strictly adhere to the following format without any deviations or additional information.

Context: Normal operating range:
{context}

Fault Models:
- Battery:
  - OK: Working as expected.
  - Empty: Provides no voltage (Battery Voltage = 0).

- H-Bridge:
  - OK: Working as expected.
  - F1: Always drives the motor forward (Mode = 1).
  - F2: Always drives the motor backward (Mode = 2).
  - F3: Always stops the motor (Mode = 0).

- Motor:
  - OK: Working as expected.
  - Broken: No current flows through it.
  - Short: High current with no voltage drop.
  - F1: 1/3 of the resistance and inductance is missing.
  - F2: 2/3 of the resistance and inductance is missing.

- Load:
  - OK: Working as expected.
  - OK_N: Load applied in the opposite direction.
  - Empty: No load applied.
  - F1: Load increased by 1.5.
  - F2: Only 0.5 of the load applied.

**Note:** 
- H-Bridge Mode 0 should correspond to the F3 fault (always stops the motor) unless otherwise indicated by additional context.
- If Battery is detected as "Empty," re-evaluate the plausibility of other components being "OK."
- If H-Bridge is in mode 0, ensure that the Motor's status is consistent with this condition.

Question: Based on the context and fault models, do the following test measurements indicate a specific fault? You must evaluate each component (Battery, H-Bridge, Motor, Load) individually and provide a fault assessment for each. If no fault is detected in a component, state "OK" for that component. If uncertain, state "uncertain."

Test Measurements:
{measurement}

Check *very carefully* with the given ranges and fault models before giving your rating. Do not infer faults not explicitly indicated by the data. **Reassess all other components if a major fault is detected in one component to ensure overall consistency.**

**Your response must be strictly in the following format:**

Answer:
- Battery: (e.g., Battery Empty, OK, uncertain)
- H-Bridge: (e.g., H-Bridge F2, OK, uncertain)
- Motor: (e.g., Motor Short, OK, uncertain)
- Load: (e.g., Load F1, OK, uncertain)
- Explanation: (Concise explanation using fault models and emphasizing the consistency of the fault type with the provided context)
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question", "measurement"],
    template=template2,
)

retriever = qdrant.as_retriever(search_type=args["search_strategy"], search_kwargs={"k": args["num_of_results"]})
chain = load_qa_chain(
        llm, 
        chain_type="stuff", 
        prompt=QA_CHAIN_PROMPT,
        verbose=True,
    )

results = {}
reason = {}

lines = range(START, END)

#logger.info(f"Model: {model}")
    
for i in tqdm(lines):
    measurement = data[i].page_content.strip()
    question = f"""Do the following measurements contain anomalies? Yes or no?"""

    docs = retriever.get_relevant_documents(measurement, return_source_documents=False)
    logger.info(f"Line {i}: {measurement}")
    logger.info(f"Retrieved {len(docs)} documents")
    logger.info(f"Question: {question}")
    result = chain.invoke({"input_documents": docs, "question": question, "measurement": measurement}, return_only_outputs=True)
    
    answer = result["output_text"].strip()
    #clear_output(wait=True)
    logger.info(f"Line {i}: {answer}")
    
    reason[i] = answer
    results[i] = False

    if "yes" in answer.lower():
        results[i] = True

df = pd.DataFrame(data={"line": lines, 
                        "model": [model for i in lines],
                        "anomaly": [results[i] for i in lines], 
                        "num_of_results": [args["num_of_results"] for i in lines],
                        "search_strategy": [args["search_strategy"] for i in lines],
                        "reason": [reason[i].strip() for i in lines],
                    })
df.to_csv(f"./../results/{COLLECTION_NAME}/results_from_{START}_to_{END}_{args['search_strategy']}_{args['num_of_results']}_{model}.csv", index=False)
