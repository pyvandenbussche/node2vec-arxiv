# node2vec-arxiv

[![Documentation Status](https://img.shields.io/badge/Blog-link_to_the_post-brightgreen.svg)](http://pyvandenbussche.info/2019/node2vec-and-arxiv-data/)

Experiment using node2vec on arXiv papers metadata. 

## Installation

### Prerequisites

* Python ≥ 3.6

### Provision a Virtual Environment

Create and activate a virtual environment (conda)

```
conda create --name py36_node2vec-arxiv python=3.6
source activate py36_node2vec-arxiv
```

If `pip` is configured in your conda environment, 
install dependencies from within the project root directory
```
pip install -r requirements.txt
``` 

## Get ArXiv dataset

The dataset used in this repository should be [downloaded from Kaggle](https://www.kaggle.com/neelshah18/arxivdataset)

Create a folder `data` from within the project root directory.
Place the downloaded file `arxivData.json` in the `data` folder.

## Running the code

Now that the environment is setup and the dataset is available, you can run the code using the following command:
```bash
python main.py 
```
This will by default use the `arxivData.json` file as input and generate in the same `data` folder the following embedding files:

- **kg_node2vec_embed.emb**: the embedding file with as first column the `node id` followed by the vector dimensions
- **kg_node2vec_label.tsv**: a mapping of `node id` to `node label`

To simplify the visualisation we output as well embeddings and labels compliant with tensorflow projector tool. Note that we **filter only to Author nodes** for the purpose of the blog post.
- **kg_node2vec_tf_proj.tsv**: an embedding file compliant with tensorflow project format (vectors without label nor id)
- **kg_node2vec_label.tsv**: an label file compliant with tensorflow project format

## Visualising the embeddings
Use [Tensorflow projector](https://projector.tensorflow.org/) to visualise the embeddings. 
You can load the data (embedding and label).
