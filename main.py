#from utils.download_files import download_json_from_zenodo
from gcr.processors import GCRProcessAgent
from utils.preprocess_pm4py import get_docs_from_pm4py, to_langchain_docs
from utils.graph_utils import ocel_to_graph_with_pm4py, load_graphml_to_networkx, build_vocabularies_from_local_graph, build_global_context_from_ocel
from eval.generate_eval_dataset import build_all_datasets
from rag.rag import get_retriever, get_retriever_from_db, create_rag_chain
from graphrag.graphrag import perform_local_search
from gcr.gcr import build_trie_from_path_strings, linearize_path, build_trie_from_ocel, extract_paths, collect_unique_path_strings
from gcr.trie import ProcessTrie
import pickle
import os
from dotenv import load_dotenv


load_dotenv()
DOCS_CACHE = "cache/pm4py_docs.pkl"

if __name__ == "__main__":
    #RECORD_ID = "8412920"
    #download_json_from_zenodo(RECORD_ID, output_dir="./zenodo_json_data")
    #docs = get_docs_extensive("data/ocel2-p2p.json")

    # docs.sort(key=lambda d: d.metadata.get("id", ""))
    # for doc in docs[:10]:
    #      print(f"--- Document ID: {doc.metadata['id']} ---")
    #      print(doc) 
    #      print("\n")
    # print("Creating retriever...")

    if os.path.exists(DOCS_CACHE):
        print("Loading docs from cache...")
        with open(DOCS_CACHE, "rb") as f:
            docs = pickle.load(f)
    else:
        print("Building docs from pm4py (slow)...")
        li_docs = get_docs_from_pm4py("data/ocel2-p2p.json")
        os.makedirs("cache", exist_ok=True)
        docs = to_langchain_docs(li_docs)
        with open(DOCS_CACHE, "wb") as f:
            pickle.dump(docs, f)
        print(f"Docs cached to {DOCS_CACHE}.")

    # retriever_openai = get_retriever(docs, "./faiss_db_openai", embedding_backend="openai")
    # print("OpenAI retriever is ready.")
    retriever_minilm = get_retriever(docs, "./faiss_db_minilm", embedding_backend="minilm")
    print("MiniLM retriever is ready.")
    # retriever_e5 = get_retriever(docs, "./faiss_db_e5", embedding_backend="e5")
    # print("E5 retriever is ready.")
    # retriever_bge = get_retriever(docs, "./faiss_db_bge", embedding_backend="bge")
    # print("BGE retriever is ready.")



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
    # graph = ocel_to_graph_with_pm4py("data/ocel2-p2p.json", "global_graph.graphml")
    #graph = build_global_context_from_ocel(input_file_path="data/ocel2-p2p.json", output_file_path="global_graph.pkl")
    #graph = load_graphml_to_networkx("global_graph.graphml")
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

    # graph = load_graphml_to_networkx("test2.graphml")
    # #build_all_datasets(graph, out_prefix="eval")
        
    # agent = GCRProcessAgent("Qwen/Qwen2.5-1.5B-Instruct", graph)
    # results = agent.generate_compliant_paths("event:25958", "What happens after event event:25958? What objects are involved?")
    # print(results)
    # results = agent.generate_compliant_paths("goods receipt:383", "What are the events associated with object goods receipt:383? Describe its lifecycle.")
    # print(results)
    #results = agent.generate_compliant_paths("event:3885", "What happens after event:3885? What objects are involved?")
    #results = agent.generate_compliant_paths("event:11991", "What happens after event:11991? What objects are involved?")
    #print(results)
