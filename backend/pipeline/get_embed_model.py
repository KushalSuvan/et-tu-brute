
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

def get_embed_model(model_name = DEFAULT_MODEL):
    model = SentenceTransformer(model_name)
    return model