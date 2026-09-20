"""Elasticsearch multi-field retrieval with externally configured credentials and index."""
import os


class SparseRetriever:
    def __init__(self, index, fields):
        from elasticsearch import Elasticsearch
        host = os.environ.get("REDYNTOT_ES_URL", "http://localhost:9200")
        options = {"request_timeout": 30}
        certificate = os.environ.get('REDYNTOT_ES_CA_CERT')
        if certificate:
            options['ca_certs'] = certificate
        user = os.environ.get("REDYNTOT_ES_USER")
        password = os.environ.get("REDYNTOT_ES_PASSWORD")
        if user and password:
            options["basic_auth"] = (user, password)
        self.connection = Elasticsearch(host, **options)
        self.index = index
        if not fields or not all(isinstance(field, str) for field in fields):
            raise ValueError('BM25 fields must be explicitly configured')
        self.fields = list(fields)

    def search(self, question, k=50):
        result = self.connection.search(index=self.index, query={"multi_match": {"query": question, "fields": self.fields}}, size=k)
        return [{"title": item["_source"].get("title", ""), "text": item["_source"].get("text", ""), "paragraph_text": item["_source"].get("text", ""), "bm25_score": item["_score"]} for item in result["hits"]["hits"]]
