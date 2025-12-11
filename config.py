"""
PROVEX Configuration

Configuration for the PROVEX explainability framework.
Training-specific settings have been removed (see Kairos pipeline for training).
"""

########################################################
#
#                   Artifacts path
#
########################################################

# The directory to save all artifacts
ARTIFACT_DIR = "./artifact/"

# JSON mapping file for node id -> human readable label
NODE_MAPPING_JSON = ARTIFACT_DIR + "explanations/node_mapping.json"

# The directory with pre-computed graph embeddings (from training pipeline)
GRAPHS_DIR = ARTIFACT_DIR + "graph_embeddings/"

# The directory with trained model checkpoints
MODELS_DIR = ARTIFACT_DIR + "models/"


########################################################
#
#       Database settings (OPTIONAL)
#
#   Only needed for export_node_mapping.py to query
#   human-readable node labels from the CADETS database.
#   The dashboard works fine without these (shows node IDs).
#
########################################################

DATABASE = 'tc_cadet_dataset_db'
HOST = None  # Set to '/var/run/postgresql/' if needed
USER = 'postgres'
PASSWORD = 'password'
PORT = '5432'


########################################################
#
#               Graph semantics
#
########################################################

# Edge types used in the temporal graph
include_edge_type = [
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

# Bidirectional map: edge type <-> edge ID
rel2id = {
    1: 'EVENT_WRITE',
    'EVENT_WRITE': 1,
    2: 'EVENT_READ',
    'EVENT_READ': 2,
    3: 'EVENT_CLOSE',
    'EVENT_CLOSE': 3,
    4: 'EVENT_OPEN',
    'EVENT_OPEN': 4,
    5: 'EVENT_EXECUTE',
    'EVENT_EXECUTE': 5,
    6: 'EVENT_SENDTO',
    'EVENT_SENDTO': 6,
    7: 'EVENT_RECVFROM',
    'EVENT_RECVFROM': 7
}


########################################################
#
#           Model dimensionality (for loading)
#
########################################################

# Node Embedding Dimension
node_embedding_dim = 16

# Neighborhood Sampling Size
neighbor_size = 64
