# Java Code Analysis

## Overview
This toolkit leverages LLMs and `javac` to detect, analyze, and fix errors in Java code. It consists of multiple units working together for debugging and code analysis. The system supports iterative error correction, code decomposition, and RAG-based memory for code analysis.

## File Structure

| File Name      | Description |
|---------------|-------------|
| `code.java`   | Sample Java code for testing. |
| `compiler.py` | Uses `javac` to detect and count errors (debugging only). |
| `unit1.py`    | Integrates LLM + `javac` for iterative error fixing. |
| `unit2.py`    | Decomposes Java code for analysis. |
| `unit3.py`    | Analyzes code components using RAG-based memory. |
| `unit3_1.py`    | Analyzes code components using RAG-based memory. Builds a Semantic link for consistency. Evaluation in 5 steps including a functtion to add rubrik|
| `viewdb.py`   | Displays embeddings and stored text from the local database. |
| `requirements.txt` | Lists all required dependencies. |

## Installation Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Provide `model_name` and `api_key` as required for LLM integration.

3. Run the main analysis module:
   ```bash
   python unit3_1.py
   ```

## Notes
- Ensure `javac` is installed and accessible in the system path.
- install `Colorful Comments` by `Parth Rastogi` for improved comment readability 

