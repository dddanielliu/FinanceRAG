# Step 1: Import necessary libraries
# --------------------------------------
# Import required libraries for document retrieval, reranking, and logging setup.
import json
import logging
import os
from typing import List

import pandas as pd
from sentence_transformers import CrossEncoder

from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from financerag.tasks import (
    BaseTask,
    ConvFinQA,
    FinDER,
    FinQABench,
    FinQA,
    FinanceBench,
    MultiHiertt,
    TATQA
)

# Setup basic logging configuration to show info level messages.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskRunner:
    def __init__(self):
        self.encoder_model = SentenceTransformerEncoder(
            model_name_or_path="intfloat/e5-large-v2",
            query_prompt="query: ",
            doc_prompt="passage: ",
        )

        self.retreival_model = DenseRetrieval(model=self.encoder_model)

        self.reranker = CrossEncoderReranker(
            model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        )

    def runtask(
        self,
        task: BaseTask,
        results_dir: str = "./results",
        save_results: bool = True,
        save_original_results: bool = True,
    ):
        retrieval_result = task.retrieve(retriever=self.retreival_model)

        # Print a portion of the retrieval results to verify the output.
        print(
            f"Retrieved results for {len(retrieval_result)} queries. Here's an example of the top 5 documents for the first query:"
        )

        for q_id, result in retrieval_result.items():
            print(f"\nQuery ID: {q_id}")
            # Sort the result to print the top 5 document ID and its score
            sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

            for i, (doc_id, score) in enumerate(sorted_results[:5]):
                print(f"  Document {i + 1}: Document ID = {doc_id}, Score = {score}")

            break  # Only show the first query

        reranking_result = task.rerank(
            reranker=self.reranker,
            results=retrieval_result,
            top_k=100,  # Rerank the top 100 documents
            batch_size=32,
        )

        # Print a portion of the reranking results to verify the output.
        print(
            f"Reranking results for {len(reranking_result)} queries. Here's an example of the top 5 documents for the first query:"
        )

        for q_id, result in reranking_result.items():
            print(f"\nQuery ID: {q_id}")
            # Sort the result to print the top 5 document ID and its score
            sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

            for i, (doc_id, score) in enumerate(sorted_results[:5]):
                print(f"  Document {i + 1}: Document ID = {doc_id}, Score = {score}")

            break  # Only show the first query

        # Step 7: Save results
        # -------------------
        # Save the results to the specified output directory as a CSV file.
        output_dir = results_dir

        if save_results:
            task.save_results(output_dir=output_dir)
        if save_original_results:
            task.save_original_results(output_dir=output_dir)

        # Confirm the results have been saved.
        print(
            f"Results have been saved to {output_dir}/{task.metadata.name}/"
        )

    # def conbine_results(self):

    @staticmethod
    def format_results(results):
        formatted_string = ""
        for metric_group in results:
            
            # Format each "Metric: value" with fixed width for perfect alignment
            formatted = [
                f"{k}: {v:.5f}".ljust(20)
                for k, v in metric_group.items()
            ]

            # Print 3 per line like your output
            for i in range(0, len(formatted), 3):
                formatted_string += ("\t" + "".join(formatted[i:i+3]).rstrip()) + "\n"
        
        logging.debug(formatted_string)
        return formatted_string

    @staticmethod
    def evaluate(task: BaseTask, results_dir: str = "./results"):
        if isinstance(task, type):  # task should be type `BaseTask`
            task = task(load_data=False)
        df_qrels = pd.read_csv(
            os.path.join(
                task.metadata.dataset["path"],
                task.metadata.dataset["subset"],
                "qrels.tsv",
            ),
            sep="\t",
        )
        # qrels_dict = (
        #     df_qrels.groupby("query_id",)
        #     .apply(lambda x: dict(zip(x["corpus_id"], x["score"])))
        #     .to_dict()
        # )
        qrels_dict = {
            qid: dict(zip(g["corpus_id"], g["score"]))
            for qid, g in df_qrels.groupby("query_id")
        }
        results = (
            task.rerank_results
            if task.rerank_results
            else task.retrieve_results
            if task.retrieve_results
            else json.load(
                open(
                    os.path.join(
                        results_dir, task.metadata.name, "original_results.jsonl"
                    ),
                    "r",
                )
            )
            if os.path.exists(
                os.path.join(results_dir, task.metadata.name, "original_results.jsonl")
            )
            else None
        )

        return task.evaluate(qrels_dict, results, [1, 5, 10])

    @staticmethod
    def combine_results(tasks: List[BaseTask], results_dir: str = "./results"):
        results = pd.DataFrame()
        for task in tasks:
            if isinstance(task, type):  # task should be type `BaseTask`
                task = task(load_data=False)
            result = (
                task.rerank_results
                if task.rerank_results
                else task.retrieve_results
                if task.retrieve_results
                else pd.read_csv(
                    os.path.join(results_dir, task.metadata.name, "results.csv")
                )
                # else json.load(
                #     open(os.path.join(results_dir, task.metadata.name, "original_results.jsonl"), "r")
                # ) if os.path.exists(os.path.join(results_dir, task.metadata.name, 'original_results.jsonl'))
                # else pd.read_csv(os.path.join(results_dir, task.metadata.name, 'results.csv')).groupby('query_id')['corpus_id'].apply(list).to_dict()
                # else None
            )
            results = pd.concat([results, result], ignore_index=True)
        results.to_csv(os.path.join(results_dir, "combined_results.csv"), index=False)
    
    @staticmethod
    def save_metrics(tasks: List[BaseTask], title: str = "", results_dir: str = "./results"):
        final_string = ""
        for task in tasks:
            if isinstance(task, type):  # task should be type `BaseTask`
                task = task(load_data=False)
            formated_string = task.metadata.name + "\n"
            eval_result = TaskRunner.evaluate(task, results_dir=results_dir)
            formated_string += TaskRunner.format_results(eval_result)
            final_string += formated_string + "\n"
        if title:
            final_string = "-- " + title + " --\n\n" + final_string
        with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
            f.write(final_string)
        return final_string

    def run(self, task: BaseTask, results_dir: str = "./results"):
        self.runtask(task, results_dir=results_dir)
        eval_result = self.evaluate(task, results_dir=results_dir)
        formated_string = self.format_results(eval_result)
        print(formated_string)


if __name__ == "__main__":
    all_tasks: List[type] = [
        ConvFinQA,
        FinDER,
        FinQABench,
        FinQA,
        FinanceBench,
        MultiHiertt,
        TATQA
    ]
    runner = TaskRunner()
    for task in all_tasks:
        current_task: BaseTask = task()
        runner.run(current_task, results_dir='./results')
    runner.combine_results(tasks=all_tasks, results_dir='results')
    runner.save_metrics(tasks=all_tasks, results_dir='results')
