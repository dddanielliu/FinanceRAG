import json
import logging
import os
from typing import List

import pandas as pd
from sentence_transformers import CrossEncoder

from typing import Optional

from financerag.rerank import CrossEncoderReranker, RRFReranker
from financerag.retrieval import (
    BM25Model,
    BM25Retriever,
    DenseRetrieval,
    SentenceTransformerEncoder,
)
from financerag.tasks import (
    TATQA,
    BaseTask,
    ConvFinQA,
    FinanceBench,
    FinDER,
    FinQA,
    FinQABench,
    MultiHiertt,
)

# Setup basic logging configuration to show info level messages.
logging.basicConfig(level=logging.INFO)


import gc
import logging

import torch

from task_runner import TaskRunner

# List of the specific loggers appearing in your output
noisy_loggers = [
    'financerag.common.loader',
    'financerag.tasks.BaseTask',
    # 'sentence_transformers',
    # 'transformers'  # sentence_transformers usually relies on this
]

for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def run(task: BaseTask, results_dir: str = "./results", task_class: Optional[type] = None):
    print(f"Running {task.metadata.name}")
    if not (task_class and isinstance(task_class, type)):
        logging.error(f"Invalid task class: {task_class}")
        return

    original: BaseTask = task_class(corpus_file='corpus.jsonl', query_file='queries.jsonl')
    # Use original.queries, original.corpus to access

    extended: BaseTask = task_class(corpus_file='corpus_prep.jsonl', query_file='queries_prep.jsonl')
    # Use extended.queries, extended.corpus to access


    encoder_model = SentenceTransformerEncoder(
        model_name_or_path='intfloat/e5-large-v2',
        query_prompt='query: ',
        doc_prompt='passage: ',
    )

    dense_retrieval_model = DenseRetrieval(
        model=encoder_model
    )

    # use original queries and corpus to do dense retrieval
    task.queries = original.queries
    task.corpus = extended.corpus
    
    dense_retrieval_result = task.retrieve(dense_retrieval_model)

    # use exteneded queries and corpus to do BM25 retrieval
    task.queries = extended.queries
    task.corpus = extended.corpus

    bm25_model = BM25Model(task.corpus)
    sparse_model = BM25Retriever(
        model=bm25_model
    )

    sparse_retrieval_result = task.retrieve(sparse_model)

    retrieval_result = (dense_retrieval_result, sparse_retrieval_result)
    

    print(f"Retrieved hybrid results for {len(retrieval_result)} queries. Here's an example of the top 5 documents for the first query:")

    for q_id, result in retrieval_result[0].items():
        print(f"\nQuery ID: {q_id}")
        # Sort the result to print the top 5 document ID and its score
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

        for i, (doc_id, score) in enumerate(sorted_results[:5]):
            print(f"  Document {i + 1}: Document ID = {doc_id}, Score = {score}")

        break  # Only show the first query

    for q_id, result in retrieval_result[1].items():
        print(f"\nQuery ID: {q_id}")
        # Sort the result to print the top 5 document ID and its score
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

        for i, (doc_id, score) in enumerate(sorted_results[:5]):
            print(f"  Document {i + 1}: Document ID = {doc_id}, Score = {score}")

        break  # Only show the first query

    # Delete references
    del encoder_model
    del dense_retrieval_model
    del bm25_model
    del sparse_model
    
    # Force garbage collection (CPU RAM)
    gc.collect()

    reranker = RRFReranker()
    
    
    # Clear GPU cache (Crucial for SentenceTransformer/PyTorch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rerank_results = task.rerank(
        reranker=reranker,
        results=retrieval_result,
        top_k=100,  # Rerank the top 100 documents
        k=6
    )

    del reranker

    task.queries = original.queries
    task.corpus = original.corpus

    reranker_model_2 = CrossEncoderReranker(
        CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=True)
    )

    rerank_2_results = task.rerank(
        reranker=reranker_model_2,
        results=rerank_results,
        top_k=60,  # Rerank the top_k documents
        batch_size=32
    )

    del reranker_model_2

    task.save_results(output_dir='./results')
    task.save_original_results(output_dir='./results')

if __name__ == '__main__':
    all_tasks: List[type] = [
        ConvFinQA,
        FinDER,
        FinQABench,
        FinQA,
        FinanceBench,
        MultiHiertt,
        TATQA
    ]
    for task in all_tasks:
        current_task: BaseTask = task(load_data=False) # (corpus_file='corpus_prep.jsonl', query_file='queries_prep.jsonl')
        TaskRunner.run_custom(current_task, run, results_dir='./results', task_class=task)
        evaluate_result = TaskRunner.evaluate(task, results_dir='./results')
        print(current_task.metadata.name)
        print(TaskRunner.format_results(evaluate_result))

    TaskRunner.combine_results(tasks=all_tasks, results_dir='results')
    metrics = TaskRunner.save_metrics(tasks=all_tasks, title='dense with keyword extraction & BM25 with prep & hybrid & rerank with jina-reranker-v2-base-multilingual-reranker-base', results_dir='results')
    print(metrics)
