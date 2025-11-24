from concurrent.futures import ThreadPoolExecutor
import json
from typing import Literal
from kg_gen.kg_gen import KGGen
import pandas as pd
import os
import dspy
import typer

from experiments.wikiqa._1_download_articles import sanitize_filename


BASE_PATH = "data/wiki_qa"
OUTPUT_ARTICLES_DIR = f"{BASE_PATH}/articles"
OUTPUT_KG_DIR = f"{BASE_PATH}/kgs"
DEFAULT_WIKIPEDIA_USER_AGENT = "MyWikiQADataFetcher/1.0 (example@example.com)"


# ------
# Remove rows where the article doesn't contain the correct answer
lm = dspy.LM(
    "openai/gpt-5-nano",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=100000,
    reasoning_effort="high",
)
dspy.configure(lm=lm)


class ArticleNoAnswer(dspy.Signature):
    """
    Determine if the Answer to the question can be found in the article.
    """

    question: str = dspy.InputField()
    article_title: str = dspy.InputField()
    article: str = dspy.InputField()
    correct_answer: str = dspy.InputField()

    does_article_contain_answer: bool = dspy.OutputField(
        desc="Whether the article contains the correct answer to question"
    )


def clean_rows_article_no_response(split_name: Literal["train", "test", "validation"]):
    csv_path = os.path.join(BASE_PATH, f"{split_name}_clean.csv")

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    valid_rows = []

    def process_row(row):
        document_title = row.get("document_title")
        if document_title:
            sanitized_title = sanitize_filename(document_title)
            article_path = os.path.join(OUTPUT_ARTICLES_DIR, f"{sanitized_title}.txt")

            if os.path.exists(article_path):
                q, a = row["question"], row["answer"]
                with open(article_path, "r") as f:
                    article_text = f.read()
                predict = dspy.Predict(ArticleNoAnswer)
                try:
                    result = predict(
                        question=q,
                        article_title=document_title,
                        article=article_text,
                        correct_answer=a,
                    )
                    print(
                        f"Q: {q} -- A: {a} -- has_answer: {result.does_article_contain_answer}"
                    )
                    if result.does_article_contain_answer:
                        return row
                except Exception as e:
                    print(f"Error predicting for {document_title}: {e}")
            else:
                print(f"Article file not found for: {document_title}")
        return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        # increase if bigger openai rate limits
        results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

    # Filter out None results
    valid_rows = [row for row in results if row is not None]

    # Create a new DataFrame and save to CSV
    output_csv_path = os.path.join(BASE_PATH, f"{split_name}_clean_2.csv")
    if valid_rows:
        valid_df = pd.DataFrame(valid_rows)
        valid_df.to_csv(output_csv_path, index=False)
        print(f"Saved {len(valid_rows)} rows with valid answers to {output_csv_path}")
    else:
        print("No rows with valid answers found")


# Generate a KG from the cleaned dataset


def main(split_name: str):
    thread_count = 100
    if split_name not in ["train", "test", "validation"]:
        raise ValueError(f"Invalid split name: {split_name}")

    csv_path = os.path.join(BASE_PATH, f"{split_name}_clean_2.csv")
    df = pd.read_csv(csv_path)
    kg = KGGen()
    os.makedirs(OUTPUT_KG_DIR, exist_ok=True)

    # Track failures
    missing_files = []
    generation_errors = []

    def process_row(row_data):
        title = row_data["document_title"]
        article_path = os.path.join(
            OUTPUT_ARTICLES_DIR, f"{sanitize_filename(title)}.txt"
        )
        if os.path.exists(article_path):
            try:
                output_kg_path = os.path.join(
                    OUTPUT_KG_DIR, f"{sanitize_filename(title)}.json"
                )
                if os.path.exists(output_kg_path):
                    print(f"KG already exists for '{title}'")
                    return {"status": "skipped", "title": title}

                with open(article_path, "r") as f:
                    article = f.read()

                graph_chunked = kg.generate(
                    input_data=article,
                    model="gemini/gemini-2.5-flash-preview-04-17",
                    api_key=os.getenv("GEMINI_API_KEY"),
                    chunk_size=2048,
                )
                # print("graph_chunked", graph_chunked)

                with open(output_kg_path, "w") as f:
                    f.write(graph_chunked.model_dump_json(indent=4))
                print(f"Saved knowledge graph for '{title}' to {output_kg_path}")
                return {"status": "success", "title": title}
            except Exception as e:
                error_info = {"title": title, "error": str(e)}
                print(f"Error generating KG for '{title}': {e}")
                return {"status": "generation_error", "data": error_info}
        else:
            print(f"Article file not found for: {title}")
            return {"status": "missing_file", "data": {"title": title}}

    # Process rows based on thread count
    if thread_count <= 1:
        # Process rows sequentially without threading
        results = [process_row(row) for _, row in df.iterrows()]
    else:
        # Process rows using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

    # Process results
    for result in results:
        if result["status"] == "missing_file":
            missing_files.append(result["data"])
        elif result["status"] == "generation_error":
            generation_errors.append(result["data"])

    # Save error logs
    if missing_files:
        missing_files_path = os.path.join(
            OUTPUT_KG_DIR, f"{split_name}_missing_files.json"
        )
        with open(missing_files_path, "w") as f:
            json.dump(missing_files, f, indent=4)
        print(
            f"Saved {len(missing_files)} missing file records to {missing_files_path}"
        )

    if generation_errors:
        errors_path = os.path.join(
            OUTPUT_KG_DIR, f"{split_name}_generation_errors.json"
        )
        with open(errors_path, "w") as f:
            json.dump(generation_errors, f, indent=4)
        print(
            f"Saved {len(generation_errors)} generation error records to {errors_path}"
        )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
