#from utils.download_files import download_json_from_zenodo
#from utils.preprocess import get_docs, get_docs_extensive, convert_to_li_document
from utils.graph_utils import ocel_to_graph_with_pm4py, load_graphml_to_networkx, build_vocabularies_from_local_graph
#from rag.p2prag import get_retriever, create_rag_agent, get_retriever_from_db
from graphrag.graphrag import perform_local_search
from dotenv import load_dotenv

load_dotenv()



if __name__ == "__main__":
    #RECORD_ID = "8412920"
    #download_json_from_zenodo(RECORD_ID, output_dir="./zenodo_json_data")
    #docs = get_docs("zenodo_json_data/ocel2-p2p.json")
    #docs = get_docs_extensive("zenodo_json_data/ocel2-p2p.json")
    #print("Creating retriever...")
    #retriever = get_retriever(docs)
    #print("Retriever is ready.")

    #print("Creating RAG agent...")
    #retriever = get_retriever_from_db()
    #agent = create_rag_agent(retriever)
    #print("RAG agent is ready.")

    #query = "What is a normal process in the procure to pay system?"
    #for step in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    #    step["messages"][-1].pretty_print()

    #docs_li = [convert_to_li_document(doc) for doc in docs[:10]]

    #print(test(docs_li))

    # kg_index, storage_context = create_knowledge_graph_index(docs_li[:5])
    # query_engine = create_query_engine(kg_index, storage_context)
    # response = query_engine.query("What is a normal process in the procure to pay system?")
    # print(response)
    #graph = ocel_to_graph_with_pm4py("zenodo_json_data/ocel2-p2p.json", "p2p_graph.graphml")
    graph = load_graphml_to_networkx("p2p_graph.graphml")
    activities, object_types, qualifiers = build_vocabularies_from_local_graph(graph)
    print(activities)
    print(object_types)
    print(qualifiers)
    #print(perform_local_search(graph, "event:52", "Describe this event and the next step in the process."))

