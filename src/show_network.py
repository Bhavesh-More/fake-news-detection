import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

#  SHOW BAYESIAN NETWORK DIAGRAM

def show_network(model_path, save_path="outputs/network_graph.png"):
    """
    Load the trained Bayesian Network and display
    a clean visual diagram of nodes and edges.
    """

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Build a directed graph using networkx
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # Manually position nodes for a clean look
    pos = {
        'is_emotional'  : (0, 2),
        'title_caps'    : (1, 2),
        'has_numbers'   : (2, 2),
        'short_article' : (3, 2),
        'label'         : (1.5, 0.5),
    }

    node_colors = {
        'is_emotional'  : '#FF6B6B',   # red
        'title_caps'    : '#FFA94D',   # orange
        'has_numbers'   : '#74C0FC',   # blue
        'short_article' : '#8CE99A',   # green
        'label'         : '#CC5DE8',   # purple (output node)
    }
    colors = [node_colors[n] for n in G.nodes()]

    labels = {
        'is_emotional'  : 'Emotional\nLanguage',
        'title_caps'    : 'Excessive\nCaps',
        'has_numbers'   : 'Has\nNumbers',
        'short_article' : 'Short\nArticle',
        'label'         : 'FAKE / REAL\n(Output)',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=3000,
        ax=ax
    )
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9,
        font_weight='bold',
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#495057',
        arrows=True,
        arrowsize=25,
        arrowstyle='-|>',
        width=2,
        ax=ax
    )

    # Titles
    ax.set_title(
        "Bayesian Network — Fake News Detector\n"
        "Feature Nodes → Output Node (label)",
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Legend patches
    legend_items = [
        mpatches.Patch(color='#FF6B6B', label='Emotional Language'),
        mpatches.Patch(color='#FFA94D', label='Excessive Caps'),
        mpatches.Patch(color='#74C0FC', label='Has Numbers'),
        mpatches.Patch(color='#8CE99A', label='Short Article'),
        mpatches.Patch(color='#CC5DE8', label='Output: Fake/Real'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)
    ax.axis('off')

    plt.tight_layout()

    # Save to outputs
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Network graph saved to: {save_path}")

    # Show on screen
    plt.show()
    print("Network graph displayed")
    print()