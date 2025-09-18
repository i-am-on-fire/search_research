import arxiv
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import networkx as nx
from tqdm import tqdm
import plotly.graph_objs as go
from IPython.display import display
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
# 3) Parameters
QUERY = "legal judgement prediction"    # change category or search query; e.g., "cat:cs.CL" or "deep learning"
N_PAPERS = 20        # number of papers to fetch (50-500 for MVP)
EMBED_MODEL = "all-mpnet-base-v2"  # good general SBERT; replace with SPECTER if you have it



# 4) Fetch arXiv metadata
def fetch_arxiv(query, max_results=200):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    rows = []
    for r in search.results():
        rows.append({
            "id": r.get_short_id(),
            "title": r.title,
            "abstract": r.summary.replace("\n", " "),
            "authors": ", ".join([a.name for a in r.authors]),
            "published": r.published,
            "pdf_url": r.pdf_url
        })
    return pd.DataFrame(rows)

print("Fetching metadata from arXiv...")
df = fetch_arxiv(QUERY, N_PAPERS)
print("Fetched", len(df), "papers")
display(df.head())

# 5) Make text for embedding
texts = (df['title'] + ". " + df['abstract']).tolist()

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

#load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
model.set_active_adapters("specter2")

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def embed_papers(papers):
    """
    papers: list of dicts with 'title' and 'abstract'
    returns: numpy array [num_papers, hidden_dim]
    """
    # concatenate title and abstract
    text_batch = [
        d["title"] + tokenizer.sep_token + (d.get("abstract") or "")
        for d in papers
    ]

    # tokenize
    inputs = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # CLS embedding = first token representation
        embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings.cpu().numpy()


embeddings = embed_papers([{"title": t.split(". ")[0], "abstract": t.split(". ")[1]} for t in texts])
embeddings = embeddings   # convert to numpy
embeddings = np.ascontiguousarray(embeddings)  # make it C-contiguous

# # 6) Compute embeddings
# print("Loading embedding model:", EMBED_MODEL)
# model = SentenceTransformer(EMBED_MODEL)
# embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# 7) Build FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # use inner product on normalized vectors for cosine
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 8) Helper to get top-k neighbors for a paper index
def topk_neighbors(index_vec, k=10):
    vec = index_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    D, I = index.search(vec, k+1)  # +1 because first is itself
    return I[0].tolist(), D[0].tolist()

# 9) Build graph: connect each node to its top-k neighbors
K = 8
G = nx.Graph()
for i, row in df.iterrows():
    G.add_node(i, title=row['title'], authors=row['authors'], published=str(row['published']))

# add edges
for i in range(len(df)):
    neigh_idx, scores = topk_neighbors(embeddings[i], k=K)
    for nid, sc in zip(neigh_idx[1:], scores[1:]):  # skip itself
        G.add_edge(i, int(nid), weight=float(sc))

print("Graph built. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

# 10) Choose a seed paper (by index) or find by title
seed_index = 0  # change or set by searching df.title
print("Seed:", df.loc[seed_index, 'title'])

# 11) Build subgraph around seed (radius or top-n by score)
# We'll take seed + neighbors + neighbors-of-neighbors up to depth 2
nodes_level1 = set([seed_index])

# collect direct neighbors
for nbr, edge_attrs in G[seed_index].items():
    print("DEBUG neighbor ID:", nbr, "edge attrs:", edge_attrs)
    nodes_level1.add(int(nbr))

# expand to neighbors-of-neighbors
sub_nodes = set([seed_index])
sub_nodes |= set(G.neighbors(seed_index))
for n in list(sub_nodes):
    sub_nodes |= set(G.neighbors(n))

# extract subgraph
subG = G.subgraph(sub_nodes).copy()


print("Subgraph nodes:", subG.number_of_nodes(), "edges:", subG.number_of_edges())

# 12) Plot subgraph with Plotly
pos = nx.spring_layout(subG, seed=42, k=0.5)  # positions for all nodes

edge_x = []
edge_y = []
for edge in subG.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text = []
for n in subG.nodes():
    x, y = pos[n]
    node_x.append(x)
    node_y.append(y)
    title = df.loc[n, 'title']
    authors = df.loc[n, 'authors']
    node_text.append(f"{title}<br>{authors}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    textposition="bottom center",
    hoverinfo='text',
    marker=dict(size=12),
    text=[df.loc[n,'id'] for n in subG.nodes()],
    hovertext=node_text
)

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=f"Paper graph around seed: {df.loc[seed_index,'id']}",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40)
             ))
fig.write_html("graph.html")
print("Graph saved to graph.html")