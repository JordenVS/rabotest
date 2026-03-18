#from utils.download_files import download_json_from_zenodo
from utils.preprocess import get_docs, get_docs_extensive, convert_to_li_document
from gcr.processors import GCRProcessAgent
from gcr.processors2 import GCRProcessAgent as GCRProcessAgent2
from utils.preprocess_pm4py import get_docs_from_pm4py
from utils.graph_utils import ocel_to_graph_with_pm4py, load_graphml_to_networkx, build_vocabularies_from_local_graph
from utils.generate_eval_dataset import build_all_datasets
from rag.p2prag import get_retriever, create_rag_agent, get_retriever_from_db
#from graphrag.graphrag import perform_local_search
#from gcr.gcr import build_trie_from_path_strings, linearize_path, build_trie_from_ocel, extract_paths, collect_unique_path_strings
from gcr.logit_processor import TrieConstrainedLogitsProcessor
from gcr.trie import ProcessTrie
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

load_dotenv()

if __name__ == "__main__":
    #RECORD_ID = "8412920"
    #download_json_from_zenodo(RECORD_ID, output_dir="./zenodo_json_data")
    #docs = get_docs("zenodo_json_data/ocel2-p2p.json")
    #docs = get_docs_extensive("data/ocel2-p2p.json")

    # docs.sort(key=lambda d: d.metadata.get("id", ""))
    # for doc in docs[:10]:
    #      print(f"--- Document ID: {doc.metadata['id']} ---")
    #      print(doc) 
    #      print("\n")
    # print("Creating retriever...")
    # retriever = get_retriever(docs, "./faiss_db_ext")
    # print("Retriever is ready.")

    #docs = get_docs_from_pm4py("data/ocel2-p2p.json")
    #docs.sort(key=lambda d: d.metadata.get("id", ""))
    #for doc in docs[:10]:
    #     print(f"--- Document ID: {doc.metadata['id']} ---")
    #     print(doc.text) 
    #     print("\n")
    # print("Creating RAG agent...")
    #retriever = get_retriever(docs, "./faiss_db_pm4py")
    #retriever = get_retriever_from_db("./faiss_db_pm4py")
    # agent = create_rag_agent(retriever)
    # print("RAG agent is ready.")

    # query = "What is a normal process in the procure to pay system?"
    # query = "What are the events associated with object purchase_order:587? Describe its lifecycle."
    # for step in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    #      step["messages"][-1].pretty_print()

    # docs_li = [convert_to_li_document(doc) for doc in docs[:10]]

    # print(test(docs_li))

    # kg_index, storage_context = create_knowledge_graph_index(docs_li[:5])
    # query_engine = create_query_engine(kg_index, storage_context)
    # response = query_engine.query("What is a normal process in the procure to pay system?")
    # print(response)
    # graph = ocel_to_graph_with_pm4py("data/ocel2-p2p.json", "test2.graphml")
    # graph = load_graphml_to_networkx("test2.graphml")
    # #activities, object_types, qualifiers = build_vocabularies_from_local_graph(graph)
    # paths = extract_paths(graph, "event:14389", max_depth=2) 
    # linearized_paths = [linearize_path(ps, graph) for ps in paths]
    # for lp in linearized_paths:
    #    print(lp)
    #print(perform_local_search(graph, "event:52", "Describe this event and the next step in the process."))

    #graph = load_graphml_to_networkx("test2.graphml")

    # 1) Prepare your tokenizer & model
    #model_name = "meta-llama/Llama-3.1-8B"   # choose the model you will use
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto",
    #     tie_word_embeddings=False
    # )

    # graph = load_graphml_to_networkx("test2.graphml")

    # #path_strings = collect_unique_path_strings(graph, ["event:52"], max_depth=3)
    # path_strings = collect_unique_path_strings(graph, ["goods receipt:74"], max_depth=3)
    # print("First unique path strings collected for GCR:")
    # for ps in path_strings[:5]:  # Print first 5 path strings
    #     print(ps)

    # trie = build_trie_from_path_strings(path_strings, tokenizer)
    # print("Trie built")
    # lp = LogitsProcessorList([TrieConstrainedLogitsProcessor(trie)])
    # print("LogitsProcessor ready")

    # # 5) Run constrained decoding
    # prompt = (
    #     "Generate a valid process path from creating a purchase order to invoice creation, then briefly explain the steps."
    # )
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print("Input prompt tokens:", inputs["input_ids"].shape)

    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=128,
    #     do_sample=False,          # start with greedy for debugging
    #     logits_processor=lp       # <-- GCR constraint hook
    # )
    # print("Output tokens:", outputs.shape)

    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # build_all_datasets(graph, out_prefix="eval")
        
    graph = load_graphml_to_networkx("test2.graphml")
    agent = GCRProcessAgent2("Qwen/Qwen2.5-1.5B-Instruct", graph)
    # #agent = GCRProcessAgent("Qwen/Qwen2.5-7B-Instruct", graph)
    #results = agent.generate_compliant_paths("event:52", "What happens after event:52?")
    #results = agent.generate_compliant_paths("event:3885", "What happens after event:3885? What objects are involved?")
    results = agent.generate_compliant_paths("event:11991", "What happens after event:11991? What objects are involved?")
    print(results)
