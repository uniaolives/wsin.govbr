import networkx as nx
import numpy as np

def simulate_apoptotic_network():
    """
    Simulates the Apoptotic Algorithm as a Boolean Graph.
    The DISC (Death-Inducing Signaling Complex) acts as a geometric AND gate.
    """
    G = nx.DiGraph()

    # Nodes representing the DISC components and the cascade
    nodes = [
        ("Death_Ligand", {"type": "input"}),
        ("Death_Receptor", {"type": "sensor"}),
        ("FADD", {"type": "adaptor"}),
        ("Caspase_10", {"type": "initiator"}),
        ("Caspase_8", {"type": "initiator"}),
        ("DISC_Complex", {"type": "gate"}),
        ("Caspase_3", {"type": "effector"}),
        ("Caspase_7", {"type": "effector"}),
        ("APOPTOSIS", {"type": "output"})
    ]
    G.add_nodes_from(nodes)

    # Edges representing signaling pathways
    edges = [
        ("Death_Ligand", "Death_Receptor"),
        ("Death_Receptor", "FADD"),
        ("FADD", "DISC_Complex"),
        ("Caspase_10", "DISC_Complex"),
        ("Caspase_8", "DISC_Complex"),
        ("DISC_Complex", "Caspase_3"),
        ("DISC_Complex", "Caspase_7"),
        ("Caspase_3", "APOPTOSIS"),
        ("Caspase_7", "APOPTOSIS")
    ]
    G.add_edges_from(edges)

    print("--- Apoptotic Algorithm: Network Topology ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Simulation logic: Boolean activation
    # DISC_Complex activates only if FADD AND (Caspase_8 OR Caspase_10) are present
    # We'll simulate a scenario where Caspase_10 is the specific trigger

    state = {node: False for node, _ in nodes}
    state["Death_Ligand"] = True
    state["Caspase_10"] = True # Specific Arkhe(n) trigger

    # Propagation
    steps = [
        ["Death_Receptor"],
        ["FADD"],
        ["DISC_Complex"],
        ["Caspase_3", "Caspase_7"],
        ["APOPTOSIS"]
    ]

    print("\n--- Execution Trace ---")
    for step in steps:
        for node in step:
            if node == "Death_Receptor":
                state[node] = state["Death_Ligand"]
            elif node == "FADD":
                state[node] = state["Death_Receptor"]
            elif node == "DISC_Complex":
                # Geometric AND gate logic
                state[node] = state["FADD"] and (state["Caspase_8"] or state["Caspase_10"])
            elif node in ["Caspase_3", "Caspase_7"]:
                state[node] = state["DISC_Complex"]
            elif node == "APOPTOSIS":
                state[node] = state["Caspase_3"] or state["Caspase_7"]

            status = "ACTIVE" if state[node] else "INACTIVE"
            print(f"Node {node:15}: {status}")

    if state["APOPTOSIS"]:
        print("\n[SUCCESS] Apoptotic Reset Triggered: System Integrity Preserved via Ordered Dissolution.")
    else:
        print("\n[STALL] Admissibility Failure: DISC Complex not formed. System in Stasis.")

if __name__ == "__main__":
    simulate_apoptotic_network()
