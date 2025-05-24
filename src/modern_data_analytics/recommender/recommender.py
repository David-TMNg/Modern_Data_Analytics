import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from modern_data_analytics.config import EMBEDDING_MODEL_NAME


class Recommender:
    def __init__(self):
        """
        Initialise recommender object
        """
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.project_ids = None
        self._project_embeddings = None

    @property
    def project_embeddings(self) -> np.ndarray:
        """
        Encapsulated property method to access project embeddings
        """
        if self._project_embeddings is None:
            logger.error("There is no project embeddings. Embeddings must be loaded or obtained from train method")
        return self._project_embeddings

    def load_pretrained_project_embeddings(self, project_ids: list[int], project_embeddings_path: str):
        """
        load pretrained project embeddings as numpy binary (.npy)

        Args:
            project_ids (list): list of project ids corresponding to project embeddings in the numpy binary
            project_embeddings_path (str): file path string of the project embeddings numpy binary

        """
        self.project_ids = project_ids
        self._project_embeddings = np.load(project_embeddings_path)

    def train(self, project_ids: list[int], project_objectives: list[str]):
        """
        Get the embeddings of the project objectives from SentenceTransformer

        Args
            project_ids (list): list of project ids corresponding to project_objectives list
            project_objects: list of project objective strings in the order of the supplied project_ids
        """
        self.project_ids = project_ids
        self._project_embeddings = self.model.encode(project_objectives, show_progress_bar=True)

    def get_top_matches(self, proposal_text: str, top_n: int = 10) -> list[tuple[int, float]]:
        """
        Given a research proposal, return a list of (projectID, similarity score) tuple
        for the top-N most similar Horizon projects.

        Args:
            proposal_text (str): String of the research proposal
            top_n (int): Number of most similar projects to return

        Return:
            list of tuples containing the most similar projects' ids and cosine similarity score
        """
        if self._project_embeddings is None:
            logger.error("No project embeddings for recommendation, loaded or obtained embeddings from train method")

        input_vec = self.model.encode([proposal_text])
        sims = cosine_similarity(input_vec, self._project_embeddings)[0]

        top_indices = np.argsort(sims)[::-1][:top_n]
        top_project_ids = [(self.project_ids[i], float(sims[i])) for i in top_indices]

        return top_project_ids
