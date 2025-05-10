from concurrent.futures import ThreadPoolExecutor
import json
from typing import Literal
from kg_gen.kg_gen import KGGen
import pandas as pd
import os
import requests
import wikipediaapi
import re
import dspy


BASE_CSV_PATH = "tests/data/wiki_qa"
OUTPUT_ARTICLES_DIR = "tests/data/wiki_qa/articles"
OUTPUT_KG_DIR = "tests/data/wiki_qa/articles_kg"
DEFAULT_WIKIPEDIA_USER_AGENT = "MyWikiQADataFetcher/1.0 (example@example.com)"


def download_and_save_split(split_name: str, base_output_path: str):
    os.makedirs(base_output_path, exist_ok=True)
    parquet_file_name = f"{split_name}-00000-of-00001.parquet"
    hf_path = f"hf://datasets/microsoft/wiki_qa/data/{parquet_file_name}"
    csv_file_path = os.path.join(base_output_path, f"{split_name}.csv")

    print(f"Downloading {split_name} data from {hf_path}...")
    try:
        df = pd.read_parquet(hf_path)
        print(f"Saving {split_name} data to {csv_file_path}...")
        df.to_csv(csv_file_path, index=False)
        print(
            f"{split_name.capitalize()} dataset successfully downloaded and saved to {csv_file_path}"
        )
        print(f"First 5 rows of {split_name}.csv:")
        print(df.head())
    except Exception as e:
        print(f"Error downloading or saving {split_name} data: {e}")


def download_all_wiki_qa_splits():
    print("Starting download of WikiQA dataset splits (train, test, validation)...")
    splits_to_download = ["train", "test", "validation"]
    for split_name in splits_to_download:
        download_and_save_split(split_name, BASE_CSV_PATH)
    print("Finished downloading all splits.")

    # Create clean dataset with only positive examples (label == 1)
    print("Creating clean dataset with only positive examples...")
    for split_name in splits_to_download:
        csv_file_path = os.path.join(BASE_CSV_PATH, f"{split_name}.csv")
        clean_file_path = os.path.join(BASE_CSV_PATH, f"{split_name}_clean.csv")

        try:
            df = pd.read_csv(csv_file_path)
            # Filter only rows where label is 1 (positive examples)
            clean_df = df[df["label"] == 1]
            # Group by question ID and keep only one positive example per question
            clean_df = clean_df.drop_duplicates(subset=["question_id"])

            # Save the clean dataset
            clean_df.to_csv(clean_file_path, index=False)
            print(f"Clean {split_name} dataset saved to {clean_file_path}")
            print(f"Original: {len(df)} rows, Clean: {len(clean_df)} rows")
        except Exception as e:
            print(f"Error creating clean dataset for {split_name}: {e}")


def load_document_titles_from_csv(csv_file_path: str) -> set:
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        print(
            "Please ensure the CSV file exists. You might need to run the download part of this script first."
        )
        exit(1)

    if "document_title" not in df.columns:
        print(f"Error: 'document_title' column not found in {csv_file_path}.")
        exit(1)

    raw_titles = df["document_title"].dropna().astype(str)
    document_titles = {title.strip() for title in raw_titles if title.strip()}
    print(
        f"Total unique, non-empty articles titles found in {csv_file_path}: {len(document_titles)}"
    )
    return document_titles


def initialize_wikipedia_api(user_agent: str = DEFAULT_WIKIPEDIA_USER_AGENT):
    return wikipediaapi.Wikipedia(language="en", user_agent=user_agent)


def sanitize_filename(title: str) -> str:
    sanitized_title = re.sub(r"[^\w\s-]", "", title).strip()
    sanitized_title = re.sub(r"[-\s]+", "_", sanitized_title)
    sanitized_title = re.sub(r"_+", "_", sanitized_title)
    if not sanitized_title:
        sanitized_title = f"untitled_{abs(hash(title)) % (10**8)}"
    return sanitized_title


def _get_revisions(page: str):
    # TODO: couldn't continue wasting time on this
    url = f"https://api.wikimedia.org/core/v1/wikipedia/en/page/{page}/history"
    print(url)
    response = requests.get(url)
    data = response.json()
    print(json.dumps(data, indent=4))
    return None, None
    target_year = 2015
    closest_revision = None
    closest_timestamp = None
    min_time_diff = float("inf")

    if "revisions" in data:
        for revision in data["revisions"]:
            print(revision)
            if "timestamp" in revision:
                timestamp = revision["timestamp"]
                # Parse the timestamp to get the year
                year = int(timestamp.split("-")[0])
                # Calculate how close this revision is to 2016
                time_diff = abs(year - target_year)

                # If we find a revision from exactly 2016, return it immediately
                if time_diff == 0:
                    return revision["id"], timestamp

                # Otherwise keep track of the closest one
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_revision = revision["id"]
                    closest_timestamp = timestamp

    return closest_revision, closest_timestamp


def fetch_and_save_wikipedia_articles_from_titles(
    document_titles: set, wiki_api: wikipediaapi.Wikipedia, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    fetched_count = 0
    skipped_count = 0

    for title in sorted(list(document_titles)):
        try:
            page = wiki_api.page(title)

            if not page.exists():
                print(f'  Page "{title}" does not exist on Wikipedia. Skipping.')
                skipped_count += 1
                continue

            article_text = page.text
            pageid = page.pageid

            if not article_text:
                skipped_count += 1
                continue

            sanitized_title_for_filename = sanitize_filename(title)
            file_path = os.path.join(output_dir, f"{sanitized_title_for_filename}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(article_text)
            print(f'Saved version of "{title}" (pageid {pageid}) to {file_path}')
            fetched_count += 1

        except Exception as e:
            print(f'An error occurred while processing "{title}": {e}')
            skipped_count += 1
            continue

    print(f"Successfully fetched and saved {fetched_count} articles.")
    print(f"Skipped {skipped_count} articles.")


def retrieve_articles_for_split(split_name: Literal["train", "test", "validation"]):
    download_all_wiki_qa_splits()

    csv_path = os.path.join(BASE_CSV_PATH, f"{split_name}_clean.csv")
    document_titles = load_document_titles_from_csv(csv_path)

    if not document_titles:
        print("No document titles to process from CSV. Exiting article fetching.")
        return

    print(f"Found {len(document_titles)} document titles to process.")

    wiki_api = initialize_wikipedia_api()

    fetch_and_save_wikipedia_articles_from_titles(
        document_titles,
        wiki_api,
        OUTPUT_ARTICLES_DIR,
    )


# ------
# Remove rows where the article doesn't contain the correct answer
lm = dspy.LM(
    "gemini/gemini-2.5-flash-preview-04-17",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=1000000,
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
    csv_path = os.path.join(BASE_CSV_PATH, f"{split_name}_clean.csv")

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
    output_csv_path = os.path.join(BASE_CSV_PATH, f"{split_name}_clean_2.csv")
    if valid_rows:
        valid_df = pd.DataFrame(valid_rows)
        valid_df.to_csv(output_csv_path, index=False)
        print(f"Saved {len(valid_rows)} rows with valid answers to {output_csv_path}")
    else:
        print("No rows with valid answers found")


# Generate a KG from the cleaned dataset


def generate_kg_from_clean_dataset(split_name: Literal["train", "test", "validation"], thread_count: int = 1):
    csv_path = os.path.join(BASE_CSV_PATH, f"{split_name}_clean_2.csv")
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
                with open(article_path, "r") as f:
                    article = f.read()

                graph_chunked = kg.generate(
                    input_data=article,
                    model="gemini/gemini-2.5-flash-preview-04-17",
                    api_key=os.getenv("GEMINI_API_KEY"),
                    chunk_size=2048,
                )
                output_kg_path = os.path.join(
                    OUTPUT_KG_DIR, f"{sanitize_filename(title)}.json"
                )

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


if __name__ == "__main__":
    splits = ["train", "test", "validation"]
    # for split in splits:
    #     # TODO: cache if files already exist, don't execute
    #     retrieve_articles_for_split(split)
    #     clean_rows_article_no_response(split)

    for split in ["test"]:
        generate_kg_from_clean_dataset(split)
