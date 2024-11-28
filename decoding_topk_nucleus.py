import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap

from torch import cuda

cuda.empty_cache()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



model_name = "meta-llama/Llama-3.2-1B-Instruct"

device_map = {"": 1}

# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    attn_implementation="eager",
)

model.eval()


input_text = "The names of daughters of Barak Obama are "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:1")
#attention_mask = torch.ones_like(input_ids).to("cuda")

outputs = model.generate(input_ids, max_length=len(input_ids.squeeze())+5)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

def get_log_prob(logits, token_id):
    # Compute the softmax of the logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities)
    
    # Get the log probability of the token
    token_log_probability = log_probabilities[token_id].item()
    return token_log_probability

def greedy_search(input_ids, node, length=5):
    if length == 0:
        return input_ids

    outputs = model(input_ids)
    predictions = outputs.logits

    # Get the predicted next sub-word (here we use top-k search)
    logits = predictions[0, -1, :]
    token_id = torch.argmax(logits).unsqueeze(0)

    # Compute the score of the predicted token
    token_score = get_log_prob(logits, token_id)

    # Add the predicted token to the list of input ids
    new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)

    # Add node and edge to graph
    next_token = tokenizer.decode(token_id, skip_special_tokens=True)
    current_node = list(graph.successors(node))[0]
    graph.nodes[current_node]['tokenscore'] = np.exp(token_score) * 100
    graph.nodes[current_node]['token'] = next_token + f"_{length}"

    # Recursive call
    input_ids = greedy_search(new_input_ids, current_node, length-1)
    
    return input_ids

def greedy_search_non_recursive(input_ids, node, length=5):
    current_input_ids = input_ids
    current_node = node

    for _ in range(length):
        outputs = model(current_input_ids)
        predictions = outputs.logits

        # Get the predicted next sub-word
        logits = predictions[0, -1, :]
        token_id = torch.argmax(logits).unsqueeze(0)

        # Compute the score of the predicted token
        token_score = get_log_prob(logits, token_id)

        # Add the predicted token to the list of input ids
        current_input_ids = torch.cat([current_input_ids, token_id.unsqueeze(0)], dim=-1)

        # Add node and edge to graph
        next_token = tokenizer.decode(token_id, skip_special_tokens=True)
        current_node = list(graph.successors(current_node))[0]
        graph.nodes[current_node]['tokenscore'] = np.exp(token_score) * 100
        graph.nodes[current_node]['token'] = next_token + f"_{length}"

    return current_input_ids






def plot_graph(graph, length, beams, score):
    fig, ax = plt.subplots(figsize=(3+1.2*beams**length, max(5, 2+length)), dpi=300, facecolor='white')

    # Create positions for each node
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    # Normalize the colors along the range of token scores
    if score == 'token':
        scores = [data['tokenscore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    elif score == 'sequence':
        scores = [data['sequencescore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    vmin = min(scores)
    vmax = max(scores)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "y", "g"], N=256) 

    # Draw the nodes
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_shape='o', alpha=1, linewidths=4, 
                            node_color=scores, cmap=cmap)

    # Draw the edges
    nx.draw_networkx_edges(graph, pos)

    # Draw the labels
    if score == 'token':
        labels = {node: data['token'].split('_')[0] + f"\n{data['tokenscore']:.2f}%" for node, data in graph.nodes(data=True) if data['token'] is not None}
    elif score == 'sequence':
        labels = {node: data['token'].split('_')[0] + f"\n{data['sequencescore']:.2f}" for node, data in graph.nodes(data=True) if data['token'] is not None}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
    plt.box(False)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if score == 'token':
        fig.colorbar(sm, ax=ax, orientation='vertical', pad=0, label='Token probability (%)')
    elif score == 'sequence':
        fig.colorbar(sm, ax=ax, orientation='vertical', pad=0, label='Sequence score')
    plt.savefig(f"beam{beams}_length{length}_score{score}.png")




    
def plot_prob_distribution(probabilities, next_tokens, sampling, potential_nb, total_nb=50):
    # Get top k tokens
    top_k_prob, top_k_indices = torch.topk(probabilities, total_nb)
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices.tolist()]

    # Get next tokens and their probabilities
    next_tokens_list = [tokenizer.decode([idx]) for idx in next_tokens.tolist()]
    next_token_prob = probabilities[next_tokens].tolist()

    # Create figure
    plt.figure(figsize=(0.4*total_nb, 5), dpi=300, facecolor='white')
    plt.rc('axes', axisbelow=True)
    plt.grid(axis='y', linestyle='-', alpha=0.5)
    if potential_nb < total_nb:
        plt.axvline(x=potential_nb-0.5, ls=':', color='grey', label='Sampled tokens')
    plt.bar(top_k_tokens, top_k_prob.tolist(), color='blue')
    plt.bar(next_tokens_list, next_token_prob, color='red', label='Selected tokens')
    plt.xticks(rotation=45, ha='right', va='top')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if sampling == 'top_k':
        plt.title('Probability distribution of predicted tokens with top-k sampling')
    elif sampling == 'nucleus':
        plt.title('Probability distribution of predicted tokens with nucleus sampling')
    plt.legend()
    plt.savefig(f'{sampling}_{time.time()}.png', dpi=300)
    plt.close()


def greedy_sampling(logits, beams):
    return torch.topk(logits, beams).indices


def top_k_sampling(logits, temperature, top_k, beams, plot=True):
    assert top_k >= 1
    assert beams <= top_k

    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    new_logits = torch.clone(logits)
    new_logits[indices_to_remove] = float('-inf')

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(new_logits / temperature, dim=-1)

    # Sample n tokens from the resulting distribution
    next_tokens = torch.multinomial(probabilities, beams)

    # Plot distribution
    if plot:
        total_prob = torch.nn.functional.softmax(logits / temperature, dim=-1)
        plot_prob_distribution(total_prob, next_tokens, 'top_k', top_k)

    return next_tokens


def nucleus_sampling(logits, temperature, p, beams, plot=True):
    assert p > 0
    assert p <= 1

    # Sort the probabilities in descending order and compute cumulative probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
    cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

    # Create a mask for probabilities that are in the top-p
    mask = cumulative_probabilities < p

    # If there's not n index where cumulative_probabilities < p, we use the top n tokens instead
    if mask.sum() > beams:
        top_p_index_to_keep = torch.where(mask)[0][-1].detach().cpu().tolist()
    else:
        top_p_index_to_keep = beams

    # Only keep top-p indices
    indices_to_remove = sorted_indices[top_p_index_to_keep:]
    sorted_logits[indices_to_remove] = float('-inf')

    # Sample n tokens from the resulting distribution
    probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
    next_tokens = torch.multinomial(probabilities, beams)

    # Plot distribution
    if plot:
        total_prob = torch.nn.functional.softmax(logits / temperature, dim=-1)
        plot_prob_distribution(total_prob, next_tokens, 'nucleus', top_p_index_to_keep)
        

    return next_tokens


def beam_search(input_ids, graph, node, length, beams,temperature:float = 0.1, top_K:int | None = None, p:float | None = None):

    if top_K is None and p is None:
        raise ValueError("Either top_K or p must be provided")  

    stack = [(input_ids, node, length)]

    while stack:
        current_input_ids, current_node, current_length = stack.pop()

        if current_length == 0:
            continue

        outputs = model(current_input_ids)
        predictions = outputs.logits

        # Get the predicted next sub-word (here we use top-k search)
        logits = predictions[0, -1, :]

        if top_K is not None:
            top_token_ids = top_k_sampling(logits, temperature, top_K,  beams)
        elif p is not None:
            top_token_ids = nucleus_sampling(logits, temperature, p, beams)
        else:
            top_token_ids = greedy_sampling(logits, beams)

        for j, token_id in enumerate(top_token_ids):
            # Compute the score of the predicted token
            token_score = get_log_prob(logits, token_id)
            cumulative_score = graph.nodes[current_node]['cumscore'] + token_score

            # Add the predicted token to the list of input ids
            new_input_ids = torch.cat([current_input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

            # Add node and edge to graph
            token = tokenizer.decode(token_id, skip_special_tokens=True)
            next_node = list(graph.successors(current_node))[j]
            graph.nodes[next_node]['tokenscore'] = np.exp(token_score) * 100
            graph.nodes[next_node]['cumscore'] = cumulative_score
            graph.nodes[next_node]['sequencescore'] = 1/(len(new_input_ids.squeeze())) * cumulative_score
            graph.nodes[next_node]['token'] = token + f"_{current_length}_{j}"

            # Add the next state to the stack
            stack.append((new_input_ids, next_node, current_length - 1))

# Explanation:
# This function performs a non-recursive beam search for sequence generation using a stack to manage the search state.
# The stack is initialized with the starting input_ids, node, and length. The function iteratively processes each state
# by popping it from the stack. For each state, it generates predictions using the model and selects the top tokens
# based on the specified sampling method (greedy, top-k, or nucleus). It then updates the graph with the token scores
# and sequences, and pushes the next states onto the stack for further exploration. The use of a stack allows the function
# to avoid recursion, which can be beneficial for managing memory usage and avoiding recursion depth limits.



def build_graph(length, beams):
    # Create a balanced tree with height 'length' and branching factor 'k'
    graph = nx.balanced_tree(beams, length, create_using=nx.DiGraph())

    # Add 'tokenscore', 'cumscore', and 'token' attributes to each node
    for node in graph.nodes:
        graph.nodes[node]['tokenscore'] = 100
        graph.nodes[node]['cumscore'] = 0
        graph.nodes[node]['sequencescore'] = 0
        graph.nodes[node]['token'] = input_text

    return graph


def get_best_sequence(G):
    """
    This function identifies the best sequence of tokens in a directed graph based on their sequence scores.

    The function operates as follows:
    1. It first identifies all the leaf nodes in the graph, which are nodes with no outgoing edges.
    2. Among these leaf nodes, it finds the one with the highest sequence score ('sequencescore' attribute).
    3. It then traces the path from this highest-scoring leaf node back to the root node (node 0).
    4. The function constructs a sequence by concatenating the 'token' attributes of the nodes along this path.
    5. Finally, it returns this sequence along with the maximum sequence score.

    Parameters:
    G (networkx.DiGraph): A directed graph where each node has 'token' and 'sequencescore' attributes.

    Returns:
    tuple: A tuple containing the best sequence as a string and the maximum sequence score as a float.
    """
    # Create a list of leaf nodes
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]

    # Get the leaf node with the highest cumscore
    max_score_node = None
    max_score = float('-inf')
    for node in leaf_nodes:
        if G.nodes[node]['sequencescore'] > max_score:
            max_score = G.nodes[node]['sequencescore']
            max_score_node = node

    # Retrieve the sequence of nodes from this leaf node to the root node in a list
    path = nx.shortest_path(G, source=0, target=max_score_node)

    # Return the string of token attributes of this sequence
    sequence = "".join([G.nodes[node]['token'].split('_')[0] for node in path])
    
    return sequence, max_score




def improved_plot_graph(graph, length, beams, score, prfix:str = "im"):
    fig, ax = plt.subplots(figsize=(4 + 1.5 * beams**length, max(6, 3 + length)), dpi=300, facecolor='white')
    
    # Node positions
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    
    # Normalize color range
    if score == 'token':
        scores = [data['tokenscore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    elif score == 'sequence':
        scores = [data['sequencescore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    vmin, vmax = min(scores), max(scores)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('rg', ["#FF4500", "#FFD700", "#32CD32"], N=256)
    
    # Dynamic node size
    node_sizes = [2000 + abs(data['tokenscore'] * 10) for _, data in graph.nodes(data=True)]
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_shape='o', alpha=0.9,
                           node_color=scores, cmap=cmap, linewidths=2, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.8, edge_color='gray')
    
    # Draw labels
    if score == 'token':
        labels = {node: f"{data['token'].split('_')[0]}\n{data['tokenscore']:.1f}%" for node, data in graph.nodes(data=True)}
    elif score == 'sequence':
        labels = {node: f"{data['token'].split('_')[0]}\n{data['sequencescore']:.2f}" for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color='black')
    
    # Remove frame
    ax.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Token Probability' if score == 'token' else 'Sequence Score', fontsize=10)
    
    # Show plot
    plt.tight_layout()
    plt.savefig(f"{prfix}_beam{beams}_length{length}_score{score}.png")




length = 5
beams = 2
top_K = 20
p = 0.5
temperature = 0.1

# Start generating text
graph = build_graph(length, beams)
beam_search(input_ids, graph, 0, length, beams, temperature, top_K)
sequence, max_score = get_best_sequence(graph)
print(f"Generated text: {sequence} {max_score}")
# Plot graph
improved_plot_graph(graph, length, beams, 'sequence', "top_k")
improved_plot_graph(graph, length, beams, 'token', "top_k")

graph = build_graph(length, beams)
beam_search(input_ids, graph, 0, length, beams, temperature, p=0.5)
sequence, max_score = get_best_sequence(graph)
print(f"Generated text: {sequence} {max_score}")
# Plot graph
improved_plot_graph(graph, length, beams, 'sequence', "nucleus")
improved_plot_graph(graph, length, beams, 'token', "nucleus")
