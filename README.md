# Mind's Loom: Complete Metaphor Identification with LLMs

This repository houses the code and data for our research on automatic conceptual metaphor identification using Large Language Models (LLMs) and a custom Knowledge Graph. Our work is detailed in the article submitted for publication, where we explore innovative methods for metaphor detection and analysis in textual data.

## Overview

The project utilizes a combination of keyword extraction, knowledge graph querying, and LLM-based analysis to identify and interpret conceptual metaphors in sentences. The Python script `conc-met-id.py` serves as the main pipeline for processing text data, extracting potential metaphors, and querying a Neo4j knowledge graph for relevant conceptual mappings.

## Dataset

Included in this repository are two key datasets:
- `balanced_df.csv`: A balanced dataset of sentences annotated with examples of conceptual metaphors.
- `processed_results.csv`: The output dataset containing original sentences alongside identified conceptual metaphors and their processed results.

## Code

The `conc-met-id.py` script integrates several components:
- **YAKE**: For keyword extraction from sentences.
- **Neo4j Graph Store**: To interface with the knowledge graph storing metaphorical mappings.
- **OpenAI GPT API**: For initial analysis and generation of queries based on extracted keywords, or an LLM of your choice.
- **Custom Query Engine**: To interact with the knowledge graph and refine metaphor identification based on LLM feedback.

## Dependencies

- pandas
- yake
- openai
- llama_index

## Usage

Ensure you have a running Neo4j instance and set the appropriate credentials within the script. Install the necessary Python libraries and run `conc-met-id.py` to process the `balanced_df.csv` dataset and generate the `processed_results.csv` output.

## Contributing

We welcome contributions and suggestions to improve the methodology and extend the dataset. Please refer to the contribution guidelines for more information.

## Citation

This section is still under definition.

If you use the data or methodology from this project in your work, please cite our article as follows:
```bibtex
@article{tbd,
  title={tbd},
  author={tbd},
  journal={tbd},
  year={tbd}
}
