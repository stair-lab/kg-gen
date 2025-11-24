# Running MINE

MINE now automatically loads evaluation data from Hugging Face, making it easier to run evaluations without managing local files.

## Quick Start

### 1. Run Evaluation

1. Set your OpenAI API key as an environment variable:
   - **Windows PowerShell:** `$env:OPENAI_API_KEY="your_actual_key_here"`
   - **Linux/Mac:** `export OPENAI_API_KEY="your_actual_key_here"`

2. Run the evaluation script:
   ```bash
   python _1_evaluation.py --model openai/gpt-5-nano --evaluation-model local
   ```

3. Results will be saved in `results/{model-config}/`:
   - `results_{i}.json` - Evaluation results for each essay
   - `kg_{i}.json` - Generated knowledge graph for each essay

### 2. Compare Results

Generate comprehensive comparison charts and statistics:
```bash
python _2_compare_results.py
```

This creates:
- `results/results.png` - Comprehensive comparison plot
- `results/summary.txt` - Detailed statistics and rankings
- `results/comparisons/` - Pairwise comparison plots

### 3. Interactive Visualization Dashboard

Launch the Streamlit dashboard to explore results interactively:
```bash
streamlit run _3_visualize.py
```

The dashboard provides:
- 📄 **Essay Browser** - View essay topics and content
- 🔍 **Query Analysis** - See retrieved contexts and evaluations for each query

## Data Loading

The evaluation script automatically:
- ✅ **Downloads evaluation data from Hugging Face** ([kg-gen-evaluation-answers](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-answers))
- ✅ **Falls back to local files** if Hugging Face is unavailable
- ✅ **Shows clear status messages** about data source

**Source Essays:** Available at [kg-gen-evaluation-essays](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-essays) - use these to generate your knowledge graphs.

## Local Development

If you prefer to use local files or Hugging Face is unavailable:
- Ensure [`answers.json`](answers.json) exists with the evaluation questions and answers
- The script will automatically detect and use local files as fallback