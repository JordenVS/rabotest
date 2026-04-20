from utils.download_files import download_json_from_zenodo
from utils.graph_utils2 import build_process_graphs_ocel2

if __name__ == "__main__":
    download_json_from_zenodo("8412920", output_dir="./data")

    ocel_path = "data/ocel2-p2p.json"  # Path to the downloaded OCEL file
    G_behavior, G_context = build_process_graphs_ocel2("data/ocel2-p2p.json", "graphs/behavior_graph.graphml", "graphs/context_graph.graphml")

