
# IAQ RAG Final Project

## Project Overview

This repository contains the final project for CSC 7644: Applied LLM Development. This project implements a domain specific retrieval augmented generation system for indoor air quality, ventilation, carbon dioxide, and computational fluid dynamics literature. The goal is to answer technical questions over a curated paper corpus with responses that are grounded in retrieved evidence rather than generated from model knowledge alone.

The system includes:

* PDF to text preprocessing
* light text cleaning and overlap based chunking
* OpenAI embeddings
* ChromaDB vector retrieval
* BM25 lexical retrieval
* hybrid fusion of lexical and vector results
* grounded answer generation with `gpt-4o-mini`
* prompt only baseline generation for comparison

The final system was evaluated on ten manually selected technical questions and compared against a baseline without retrieval.

## Key Features

- Answers technical questions over a curated IAQ and CFD literature corpus using RAG
- Supports PDF to text preprocessing for local document ingestion
- Combines BM25 lexical retrieval with ChromaDB vector retrieval through hybrid fusion
- Generates grounded answers using retrieved passages rather than prompt-only generation
- Provides a prompt-only baseline for comparison
- Includes a manual evaluation workflow over ten technical questions

## Tech Stack and Architecture

The system was implemented in Python and uses the OpenAI API for embeddings and grounded answer generation. The main retrieval components are BM25 for lexical search and ChromaDB for vector search. PDF preprocessing is handled with `pypdf`, and the final answer generation step uses `gpt-4o-mini`.

At a high level, the pipeline consists of:
- a preprocessing stage that converts PDFs to text
- a retrieval stage that chunks, embeds, and indexes the corpus
- a hybrid search stage that combines lexical and vector retrieval
- an answer generation stage that uses the retrieved passages as evidence
- a prompt-only baseline script used for comparison

## Repository Structure

```text
.
├── extract_pdfs_to_text.py
├── Chahardoli_Saeid_csc7644_final.py
├── baseline_qa.py
├── evaluation_IAQ_questions.csv
├── evaluation_results.csv
├── README.md
├── rag/
└── baseline/
```

## Main Files

### `extract_pdfs_to_text.py`

Reads PDF files from a folder, extracts page level text, and saves one `.txt` file per paper for downstream ingestion.

### `Chahardoli_Saeid_csc7644_final.py`

Main RAG pipeline. This script supports:

* ingestion of the local text corpus
* BM25 search
* vector search with OpenAI embeddings and ChromaDB
* hybrid retrieval
* grounded answer generation from retrieved passages

### `baseline_qa.py`

Prompt only baseline used for comparison with the RAG system.

### `evaluation_IAQ_questions.csv`

Question set used for evaluation.

### `evaluation_results.csv`

Manual scoring table comparing RAG and baseline outputs.

## Environment Setup

This project was developed in a separate Conda environment.

### Create and activate environment

```bash
conda create -n final_project python=3.11 -y
conda activate final_project
```

### Install required packages

```bash
pip install openai chromadb rank-bm25 python-dotenv pypdf
```

## API Key Setup

Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit the `.env` file to GitHub.

## Data Preparation

Place the research paper PDFs in a folder such as:

```text
papers/
```

Then run the preprocessing script:

```bash
python extract_pdfs_to_text.py --input_dir ./papers --output_dir ./data/corpus
```

This creates:

* one `.txt` file per PDF in `data/corpus`
* a metadata file for the extracted corpus

## Ingestion

After preprocessing, ingest the corpus into ChromaDB.

```bash
python Chahardoli_Saeid_csc7644_final.py ingest --data_dir ./data/corpus --db_path ./kb_v3 --collection iaq_docs_v3
```

The final version used larger chunk settings than the original class defaults in order to reduce retrieval noise. The code defaults were updated accordingly.

## Search

To inspect retrieval results only:

```bash
python Chahardoli_Saeid_csc7644_final.py search --query "How does sensor placement affect CO2 based demand controlled ventilation?" --retriever hybrid --top_k 5 --db_path ./kb_v3 --collection iaq_docs_v3
```

## Grounded Answer Generation

To generate a RAG answer:

```bash
python Chahardoli_Saeid_csc7644_final.py answer --query "What are the limitations of CO2 based ventilation design?" --retriever hybrid --top_k 5 --db_path ./kb_v3 --collection iaq_docs_v3
```
**Note:** Some local development folders such as `papers/`, `data/corpus/`, and `kb_v3/` were used during preprocessing and indexing but are not fully included in the public repository.

## Prompt Only Baseline

To generate a baseline answer without retrieval:

```bash
python baseline_qa.py --query "What are the limitations of CO2 based ventilation design?"
```

## Evaluation Procedure

The system was evaluated on ten manually selected questions covering:

* ventilation and CO2
* CFD and contaminant transport
* airborne transmission
* occupant movement
* sensor placement
* non uniform CO2 distributions
* green walls

For each question:

1. a RAG answer was generated
2. a prompt only baseline answer was generated
3. both outputs were scored manually

The scoring rubric included:

* retrieval relevance
* groundedness
* completeness
* clarity

## Summary of Results

The final RAG system generally outperformed the prompt only baseline in groundedness and domain specificity. Across the ten evaluation questions, the RAG system achieved average scores of 4.1/5 for retrieval relevance, 4.2/5 for groundedness, 4.0/5 for completeness, and 4.3/5 for clarity, while the prompt only baseline remained clearer overall but less grounded in the paper corpus.

The strongest results were observed for focused technical questions, especially:

* sensor placement in CO2 based demand controlled ventilation
* breathing zone CO2 concentration
* limitations of CO2 based ventilation design

The baseline answers were often clear and plausible, but they were more generic and were not tied to the uploaded literature. Broader questions showed a smaller performance gap because the baseline could still answer them reasonably from general technical knowledge.

## Workflow Summary

The implemented workflow is:

**PDF papers**
→ **PDF to text extraction**
→ **light cleaning and overlap based chunking**
→ **OpenAI embeddings + BM25 indexing**
→ **ChromaDB vector store**
→ **hybrid retrieval of top passages**
→ **user query + top retrieved passages**
→ **grounded answer generation with `gpt-4o-mini`**
→ **final answer shown to user**

## Notes and Limitations

This project is a proof of concept rather than a production ready literature assistant. The main limitations are:

* the corpus includes multiple related subtopics, so some broad queries retrieve mixed evidence
* PDF extraction still introduces some formatting artifacts
* the current answer citations are passage based rather than full document and page citations

## Attributions and Citations

The overall RAG pipeline was developed from the course project workflow and adapted for this final project. No external repository code was copied directly into this project. The implementation was informed by official documentation for the OpenAI API, ChromaDB, BM25, and `pypdf`.

If any code is later adapted from external sources, those sources should be cited here and in code comments where appropriate.

## GitHub Repository

https://github.com/SAeidch/iaq-rag-final-project


## Author

Saeid Chahardoli

