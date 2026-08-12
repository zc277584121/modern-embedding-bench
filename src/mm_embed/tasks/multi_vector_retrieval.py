from mm_embed.data.multi_vector_fixture import DATASET_VERSION, load_multi_vector_fixture
from mm_embed.indexes.multi_vector_exact import ExactMaxSimIndex
from mm_embed.providers.multi_vector_base import MultiVectorProvider
from mm_embed.tasks.base import EvalResult

class MultiVectorRetrievalTask:
    name = "multi_vector_retrieval"
    def __init__(self, *, dataset_version=DATASET_VERSION, top_k=10, fixture_only=True, aggregation="max_passage"):
        if dataset_version != DATASET_VERSION or not fixture_only: raise ValueError("Multi-vector task is fixture-only")
        self.top_k = top_k; self.aggregation = aggregation
    def run(self, provider, **kwargs):
        if not isinstance(provider, MultiVectorProvider): raise TypeError("Multi-vector provider required")
        docs, queries, qrels, hard_negatives, digest = load_multi_vector_fixture()
        encoded_docs = provider.encode_multi_vector_passages([d.text for d in docs], passage_ids=[d.passage_id for d in docs], document_ids=[d.document_id for d in docs])
        index = ExactMaxSimIndex(encoded_docs, aggregation=self.aggregation)
        relevant_by_query = {}
        for qrel in qrels:
            relevant_by_query.setdefault(qrel.query_id, set()).add(qrel.document_id)
        rankings=[]; recalls=[]
        for q in queries:
            result=provider.encode_multi_vector_query(q.text,item_id=q.query_id); hits=index.search(result,k=self.top_k); rankings.append({"query_id":q.query_id,"diagnostic_slice":q.diagnostic_slice,"hits":[h.__dict__ for h in hits]}); recalls.append(float(hits and hits[0].document_id in relevant_by_query[q.query_id]))
        representation = encoded_docs.embeddings.representation
        return EvalResult(self.name, encoded_docs.provider, encoded_docs.model_name, {"recall@1":sum(recalls)/len(recalls)}, {"fixture_sha256":digest,"fixture_only":True,"publish":False,"leaderboard_publish":False,"rankings":rankings,"qrels":[row.__dict__ for row in qrels],"hard_negatives":[row.__dict__ for row in hard_negatives]}, {"retrieval_kind":"multi_vector_exact_maxsim","representation_kind":"multi_vector","provider":encoded_docs.provider,"model_name":encoded_docs.model_name,"model_revision":encoded_docs.model_revision,"representation_id":representation.representation_id,"representation_identity":representation.identity,"dimensions":representation.dimensions,"query_route":provider.query_route.value,"document_route":encoded_docs.route.value,"valid_token_count":sum(encoded_docs.embeddings.valid_token_counts),"passage_count":len(docs),"document_count":len(set(d.document_id for d in docs)),"aggregation":self.aggregation,"exact_backend":index.backend,"exact":True,"index_bytes_estimate":int(encoded_docs.embeddings.values.nbytes),"encoding_latency_ms":0.0,"search_latency_ms":0.0,"peak_memory_bytes":None,"peak_vram_bytes":0})
