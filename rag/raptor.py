import umap
import numpy as np
from sklearn.mixture import GaussianMixture
import trio
from rag.utils import truncate


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    def __init__(
        self, max_cluster, llm_model, embd_model, prompt, max_token=512, threshold=0.1
    ):
        self._max_cluster = max_cluster
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._threshold = threshold
        self._prompt = prompt
        self._max_token = max_token

    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state: int):
        max_clusters = min(self._max_cluster, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters

    async def __call__(self, chunks, random_state, callback=None):
        if len(chunks) <= 1:
            return []
        chunks = [(s, a) for s, a in chunks if s and len(a) > 0]
        layers = [(0, len(chunks))]
        start, end = 0, len(chunks)

        async def summarize(ck_idx: list[int]):
            nonlocal chunks
            texts = [chunks[i][0] for i in ck_idx]
            len_per_chunk = int(
                (self._llm_model.max_length - self._max_token) / len(texts)
            )
            cluster_content = "\n".join(
                [truncate(t, max(1, len_per_chunk)) for t in texts]
            )
        labels = []
        while end - start > 1:
            embeddings = [embd for _, embd in chunks[start:end]]
            if len(embeddings) == 2:
                await summarize([start, start + 1])
                if callback:
                    callback(
                        msg="Cluster one layer: {} -> {}".format(
                            end - start, len(chunks) - end
                        )
                    )
                labels.extend([0, 0])
                layers.append((end, len(chunks)))
                start = end
                end = len(chunks)
                continue

            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors),
                n_components=min(12, len(embeddings) - 2),
                metric="cosine",
            ).fit_transform(embeddings)
            n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state)
            if n_clusters == 1:
                lbls = [0 for _ in range(len(reduced_embeddings))]
            else:
                gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
                gm.fit(reduced_embeddings)
                probs = gm.predict_proba(reduced_embeddings)
                lbls = [np.where(prob > self._threshold)[0] for prob in probs]
                lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]

            async with trio.open_nursery() as nursery:
                for c in range(n_clusters):
                    ck_idx = [i + start for i in range(len(lbls)) if lbls[i] == c]
                    assert len(ck_idx) > 0
                    nursery.start_soon(summarize, ck_idx)

            assert len(chunks) - end == n_clusters, "{} vs. {}".format(
                len(chunks) - end, n_clusters
            )
            labels.extend(lbls)
            layers.append((end, len(chunks)))
            if callback:
                callback(
                    msg="Cluster one layer: {} -> {}".format(
                        end - start, len(chunks) - end
                    )
                )
            start = end
            end = len(chunks)

        return chunks
