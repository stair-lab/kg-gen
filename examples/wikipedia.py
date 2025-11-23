from datasets import load_dataset
from kg_gen.kg_gen import KGGen
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
# import mlflow

# First, find 10 articles that meet the criteria
print("Finding articles...")
fw = load_dataset("HuggingFaceFW/finewiki", name="en", split="train", streaming=True)
articles_to_process = []

os.makedirs("examples/wikipedia", exist_ok=True)


# def setup_mlflow():
#     try:
#         mlflow.dspy.autolog()
#         mlflow.set_experiment("wikipedia")
#         mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     except Exception as e:
#         # run mlflow server in another terminal
#         print(f"Error initializing MLflow: {e}")


for article in fw:
    title = article["title"]
    text = article["text"]
    if len(text) >= 6000 and len(text) <= 6400:  # +-200 std of mean from dataset
        articles_to_process.append({"title": title, "text": text})
        print(f"Found article: {title} (length: {len(text)})")
        if len(articles_to_process) >= 10:
            break

print(f"\nFound {len(articles_to_process)} articles. Starting parallel processing...")


# Function to process a single article in a thread
def process_article(article_data):
    title = article_data["title"]
    text = article_data["text"]

    # Initialize KGGen in this thread
    kg = KGGen(
        model="openai/gpt-5-nano",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1.0,
        disable_cache=True,
    )
    # setup_mlflow()

    try:
        graph = kg.generate(input_data=text)
        token_usage = kg.extract_token_usage_from_history()
        graph.to_file(f"./examples/wikipedia/{title}.json")
        print(f"✓ Generated graph for {title}")
        return {"title": title, "token_usage": token_usage, "success": True}
    except Exception as e:
        print(f"✗ Error processing {title}: {e}")
        return {"title": title, "error": str(e), "success": False}


# Process articles in parallel using multiple threads
results = []
with ThreadPoolExecutor(max_workers=16) as executor:
    # Submit all tasks
    future_to_article = {
        executor.submit(process_article, article): article
        for article in articles_to_process
    }

    for future in as_completed(future_to_article):
        result = future.result()
        results.append(result)

with open("examples/wikipedia/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Completed!")
