import os
import sys
import nltk
import argparse
from kg_gen import KGGen

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate a knowledge graph from text input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python run_kg_extraction.py --inputtext input.txt --graphname mygraph
    python run_kg_extraction.py --inputtext data.txt --model gpt-4 --output-csv
    
The script requires a text file as input and will generate Neo4j-compatible output files.
Make sure to set the OPENAI_API_KEY environment variable before running.""")
    
    parser.add_argument('--inputtext', type=str, required=True,
                        help='Path to a text file containing input data (Required)')
    parser.add_argument('--graphname', type=str, default='kg', 
                        help='Name for the generated graph files (default: kg)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--context', type=str, default='',
                        help='Optional context to guide knowledge graph extraction')
    parser.add_argument('--output-csv', action='store_true',
                        help='Output CSV files for alternative import method')
    
    # If no arguments provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

def sanitize_for_neo4j(text):
    """Clean strings for Neo4j compliance"""
    if not text:
        return text
        
    # Handle entities starting with numbers
    if text[0].isdigit():
        text = f"Year{text}" if text.isdigit() else f"Num{text}"
    
    # Replace special characters and spaces
    replacements = {
        "'": "",
        '"': "",
        ' ': '_',
        '&': 'And',
        '.': '_',
        ',': '_',
        '-': '_',
        '+': 'Plus',
        '/': '_',
        '\\': '_',
        '(': '_',
        ')': '_',
        ':': '_',
        ';': '_',
        '@': 'At',
        '#': 'Hash',
        '%': 'Percent',
        '!': '_',
        '?': '_',
        '*': '_'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Remove any other non-alphanumeric characters
    text = ''.join(c for c in text if c.isalnum() or c == '_')
    
    # Ensure the string doesn't have consecutive underscores
    while '__' in text:
        text = text.replace('__', '_')
    
    # Trim underscores from the ends
    text = text.strip('_')
    
    return text

# --- Main Script ---
def main():
    args = parse_arguments()
    
    # --- Configuration ---
    OPENAI_MODEL_NAME = args.model
    MAX_TOKENS = 1000
    GRAPH_NAME = args.graphname
    
    # --- Prerequisites Check ---
    
    # 1. Check for OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the environment variable before running the script:")
        print("export OPENAI_API_KEY='your_api_key'")
        exit(1)
    
    # 2. Download NLTK 'punkt' data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' tokenizer found.")
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        try:
            nltk.download('punkt')
            print("NLTK 'punkt' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading NLTK 'punkt': {e}")
            print("Please try downloading manually: run 'python -m nltk.downloader punkt'")
            exit(1)
    
    # --- Initialize KGGen ---
    # KGGen handles DSPy configuration internally
    try:
        kg = KGGen(model=OPENAI_MODEL_NAME, temperature=0.0, api_key=api_key)
        print(f"KGGen initialized with model: {OPENAI_MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing KGGen: {e}")
        exit(1)
    
    # --- Input Text ---
    try:
        with open(args.inputtext, 'r', encoding='utf-8') as f:
            input_text = f.read()
        print(f"\n--- Using input from file: {args.inputtext} ---")
    except Exception as e:
        print(f"Error reading input file {args.inputtext}: {e}")
        exit(1)
    
    # Preview the first 200 characters of input
    print("\n--- Input Text Preview ---")
    preview = input_text[:200] + "..." if len(input_text) > 200 else input_text
    print(preview)
    
    # --- Knowledge Graph Extraction ---
    try:
        print("\n--- Extracting Knowledge Graph ---")
        graph = kg.generate(input_data=input_text, context=args.context)
        
        print("\n--- Results ---")
        print(f"Entities: {graph.entities}")
        print(f"Edges: {graph.edges}")
        print(f"Relations: {graph.relations}")
        
        # --- Export for Neo4j ---
        # Create Cypher statements for Neo4j
        neo4j_output_file = f"{GRAPH_NAME}_import.cypher"
        with open(neo4j_output_file, "w") as f:
            # Create nodes for each entity
            for entity in graph.entities:
                # Clean the entity name for Cypher query
                clean_entity = sanitize_for_neo4j(entity)
                original_entity = entity.replace("'", "\\'")
                f.write(f"CREATE (:{clean_entity} {{name: '{original_entity}'}})\n")
            
            # Add semicolon to separate node creation from relationship creation
            f.write(";\n\n")
            
            # Create relationships between entities
            for source, relation, target in graph.relations:
                # Clean strings for Cypher query
                clean_source = sanitize_for_neo4j(source)
                clean_relation = sanitize_for_neo4j(relation)
                clean_target = sanitize_for_neo4j(target)
                
                original_source = source.replace("'", "\\'")
                original_target = target.replace("'", "\\'")
                
                # Use property matching to ensure we get the right nodes
                f.write(f"MATCH (a:{clean_source} {{name: '{original_source}'}}), (b:{clean_target} {{name: '{original_target}'}}) CREATE (a)-[:{clean_relation.upper()}]->(b);\n")
        
        print(f"\n--- Neo4j import file created: {neo4j_output_file} ---")
        print("Run this file in Neo4j Browser to import your knowledge graph")
        
        # --- Export as CSV files (only if --output-csv flag is passed) ---
        if args.output_csv:
            # Create CSV files for alternative import method
            nodes_file = f"{GRAPH_NAME}_nodes.csv"
            edges_file = f"{GRAPH_NAME}_relationships.csv"
            
            # Create mapping of original entity names to IDs
            entity_map = {entity: i for i, entity in enumerate(graph.entities)}
            
            with open(nodes_file, "w") as f:
                f.write("id,label,name\n")
                for entity, idx in entity_map.items():
                    # Clean entity for Neo4j label but keep original name
                    clean_entity = sanitize_for_neo4j(entity)
                    f.write(f"{idx},{clean_entity},\"{entity}\"\n")
            
            with open(edges_file, "w") as f:
                f.write("source,target,type\n")
                for source, relation, target in graph.relations:
                    source_id = entity_map[source]
                    target_id = entity_map[target]
                    # Clean relation for Neo4j
                    clean_relation = sanitize_for_neo4j(relation).upper()
                    f.write(f"{source_id},{target_id},{clean_relation}\n")
            
            # --- Neo4j CSV Import Script ---
            # Create a helper script for importing the CSVs
            csv_import_file = f"{GRAPH_NAME}_csv_import.cypher"
            with open(csv_import_file, "w") as f:
                # Add comments explaining the import
                f.write("// Neo4j CSV Import Script\n")
                f.write(f"// Generated for graph: {GRAPH_NAME}\n\n")
                
                # Create indexes (optional but recommended for larger graphs)
                f.write("// Create indices for faster lookups\n")
                f.write("CREATE INDEX ON :Entity(id);\n\n")
                
                # Import nodes
                f.write("// Import nodes\n")
                f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{nodes_file}' AS row\n")
                f.write("MERGE (n:Entity {id: toInteger(row.id)})\n")
                f.write("SET n.name = row.name\n")
                f.write("SET n:`${row.label}`;\n\n")
                
                # Import relationships
                f.write("// Import relationships\n")
                f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{edges_file}' AS row\n")
                f.write("MATCH (source:Entity {id: toInteger(row.source)})\n")
                f.write("MATCH (target:Entity {id: toInteger(row.target)})\n")
                f.write("CALL apoc.create.relationship(source, row.type, {}, target) YIELD rel\n")
                f.write("RETURN count(*);\n")
            
            print(f"\n--- CSV files created: {nodes_file}, {edges_file} ---")
            print(f"--- CSV import script created: {csv_import_file} ---")
            print("These files can be used for alternative import methods")
        
    except Exception as e:
        print(f"Error generating knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
