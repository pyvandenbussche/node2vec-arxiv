import argparse
import collections
import beautifultable as bt
from gensim.models import Word2Vec
import networkx as nx
import node2vec
import numpy as np
import os
import pandas as pd
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

EMBED_FILE = "kg_node2vec_embed.emb"
LABEL_FILE = "kg_node2vec_label.tsv"

TF_EMBED_PROJ_FILE = "kg_node2vec_tf_proj.tsv"
TF_EMBED_PROJ_LABEL_FILE = "kg_node2vec_tf_proj_label.tsv"


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='data/arxivData.json', help='Input Arxiv json file')
    parser.add_argument('--outputdir', nargs='?', default='data/', help='Embeddings path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]  # convert each vertex id to a string
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)

    model.wv.save_word2vec_format(os.path.join(args.outputdir, EMBED_FILE))

    return


def load_data(input_file):

    # read the arxiv input json file
    df = pd.read_json(input_file, orient='records')

    # flatten author list names.
    # this is not the most elegant but is made to handle the variation in single/double quotes for name values:
    # "author": "[{'name': 'Luciano Serafini'}, {'name': \"Artur d'Avila Garcez\"}]",
    df['author_list'] = df['author'].apply(lambda author_str: [x.strip()[10:-2] for x in author_str[1:-1].split(",")])

     # flatten tags list
    def flatten_tags(tag_str):
        tags = tag_str[1:-1].split("{'term': '")
        tags = list(filter(None, [tag.strip()[:tag.find("'")] for tag in tags]))
        return tags
    df['tags_list'] = df['tag'].apply(flatten_tags)

    return df


def build_kg(df):
    # for each row create an edge for each
    # - paper -> author
    # - paper -> tag
    kg = []
    author_set = tag_set = set()
    cpt_hasAuthor = cpt_hasTag = 0

    for index, row in df.iterrows():
        paper_id = row['id']
        author_set.update(row['author_list'])
        tag_set.update(row['tags_list'])

        for author in row['author_list']:
            kg.append([paper_id, 'hasAuthor', author])
            cpt_hasAuthor += 1
        for tag in row['tags_list']:
            kg.append([paper_id, 'hasTag', tag])
            cpt_hasTag+=1

    kg = np.asarray(kg)

    # output KG stats
    table = bt.BeautifulTable()
    table.append_row(["# statements", kg.shape[0]])
    table.append_row(["# relation type", 2])
    table.append_row(["   # hasAuthor relation", cpt_hasAuthor])
    table.append_row(["   # hasTag relation", cpt_hasTag])
    table.append_row(["# entities of type Author", len(author_set)])
    table.append_row(["# entities of type Papers", len(df.index)])
    table.append_row(["# entities of type Tag", len(tag_set)])
    table.column_alignments[0] = bt.ALIGN_LEFT
    table.column_alignments[1] = bt.ALIGN_RIGHT
    print(table)
    return kg, author_set


def export_to_tf_projector(filter=None):
    '''
    Write output files for Tensorflow embedding projector https://projector.tensorflow.org/
    '''
    labels = np.genfromtxt(os.path.join(args.outputdir, LABEL_FILE), delimiter="\t",dtype='str', encoding="utf-8")
    np.savetxt(os.path.join(args.outputdir, TF_EMBED_PROJ_LABEL_FILE), labels[:,1], delimiter="\t", fmt="%s", encoding="utf-8")
    if filter is not None:
        mask_array = np.zeros(len(labels), dtype=bool)
        mask_array[filter] = True
        np.savetxt(os.path.join(args.outputdir, TF_EMBED_PROJ_LABEL_FILE), labels[np.array(mask_array)][:, 1],
                   delimiter="\t", fmt="%s", encoding="utf-8")

        embed = np.genfromtxt(os.path.join(args.outputdir, EMBED_FILE), delimiter=" ", skip_header=1, encoding="utf-8")
        # we need to sort the embeddings by index
        ind = np.argsort(embed[:, 0]);
        embed = embed[ind]
        np.savetxt(os.path.join(args.outputdir, TF_EMBED_PROJ_FILE), embed[np.array(mask_array)][:, 1:],
                   delimiter="\t", encoding="utf-8")
    else:
        np.savetxt(os.path.join(args.outputdir, TF_EMBED_PROJ_LABEL_FILE), labels[:, 1],delimiter="\t",
                   fmt="%s", encoding="utf-8")

        embed = np.genfromtxt(os.path.join(args.outputdir, EMBED_FILE), delimiter=" ", skip_header=1, encoding="utf-8")
        # we need to sort the embeddings by index
        ind = np.argsort(embed[:, 0]);
        embed = embed[ind]
        np.savetxt(os.path.join(args.outputdir, TF_EMBED_PROJ_FILE), embed[:, 1:], delimiter="\t", encoding="utf-8")


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print("Building Arxiv KG")
    # load data:
    df = load_data(input_file=args.input)

    # convert to a KG made of triple statements
    kg, author_set = build_kg(df)

    print("Loading the KG in Networkx")
    # create an id to subject/object label mapping
    set_nodes = set().union(kg[:,0], kg[:,2])
    # save label dictionary to file
    node_to_idx = collections.OrderedDict(zip(set_nodes, range(len(set_nodes))))
    idx_to_node = np.asarray([[v, k] for k, v in node_to_idx.items()])
    np.savetxt(os.path.join(args.outputdir, LABEL_FILE), idx_to_node, delimiter="\t", fmt="%s", encoding="utf-8")

    nx_G = nx.DiGraph()
    nx_G.add_nodes_from(range(0, len(set_nodes)))
    for s, p, o in kg:
        nx_G.add_edge(node_to_idx[s], node_to_idx[o], type=p)
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1
    G_undir = nx_G.to_undirected()

    print("Computing transition probabilities and simulating the walks")
    start_time = time.time()
    G = node2vec.Graph(G_undir, False, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    print("Learning the embeddings and writing them to file")
    learn_embeddings(walks)
    elapsed_time = time.time() - start_time
    print("Node2vec algorithm took: {}"
          .format(time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed_time))))

    # export embedding and  labels to tensorflow projector format
    print("Export the authors embeddings to TensorFlow project format")
    export_to_tf_projector(filter = [node_to_idx[author] for author in author_set])


if __name__ == "__main__":
    args = parse_args()
    main(args)
