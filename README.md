# PDF RAG Tutorial

A beginner-friendly implementation of **Retrieval-Augmented Generation
(RAG)** using Python, LangChain, OpenAI embeddings, and Chroma. This
project demonstrates how to retrieve relevant information from a PDF and
use an LLM to generate an answer based on the retrieved context.

## What is RAG?

Retrieval-Augmented Generation combines two steps:

1.  **Retrieval:** Search a knowledge source for text relevant to a
    user's question.
2.  **Generation:** Pass the retrieved text and the question to a Large
    Language Model (LLM) to produce a natural-language response.

In short, retrieval finds potentially useful evidence; generation turns
that evidence into an answer.

## Project contents

``` text
RAG-Tutorial/
├── RAG_PDF.ipynb
├── RAG_PDF_Semantic_Search_vs_LLM_Guide.md
├── Docs/
│   └── ABC_Bank_Customer_Support_FAQ_Sample.pdf
├── .gitignore
└── requirements.txt
```

Your actual filenames may differ slightly. Update this section to match
the files you commit.

## What you'll learn

-   Load and extract text from a PDF with `PyPDFLoader`.
-   Understand LangChain `Document` objects and metadata.
-   Split documents into smaller chunks with
    `RecursiveCharacterTextSplitter`.
-   Create text embeddings using `text-embedding-3-small`.
-   Store chunks and vectors in Chroma.
-   Perform semantic similarity search.
-   Pass retrieved context to an LLM with `ChatOpenAI`.
-   Understand why retrieved chunks and generated answers look
    different.
-   Improve prompts to encourage answers grounded in retrieved context.

## Technology stack

-   Python
-   Jupyter Notebook
-   LangChain
-   OpenAI chat model and embeddings
-   Chroma vector store
-   PyPDF

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/Prabhas92/RAG-Tutorial.git
cd RAG-Tutorial
```


### 2. Create and activate a virtual environment

On macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The repository includes `dependencies.txt`:

``` bash
pip install -r dependencies.txt
```


### 4. Configure your API key

Create a local `.env` file in the project root:

``` env
OPENAI_API_KEY=your_api_key_here
```

Replace the placeholder with your own API key. **Never commit `.env` or
expose your API key in notebook output.** The `.gitignore` file should
include `.env`.

### 5. Run the notebook

``` bash
jupyter notebook RAG_PDF.ipynb
```

Open the notebook and run the cells in order. Make sure the PDF path in
the notebook matches the location and name of the PDF in `Docs/`.

## How the pipeline works

``` text
PDF
 |
 v
PyPDFLoader
 |
 v
Documents (text + metadata)
 |
 v
RecursiveCharacterTextSplitter
 |
 v
Text chunks
 |
 v
OpenAI embeddings
 |
 v
Chroma vector store
 |
 v
Question -> semantic search
 |
 v
Retrieved chunks
 |
 v
Question + retrieved context -> LLM
 |
 v
Generated answer
```

## Semantic search vs. LLM response

**Semantic search** returns the document chunks most similar to a query.
It does not itself produce a conversational answer or guarantee that the
returned chunks contain the exact answer.

**The LLM response** is newly generated text based on the prompt and
context supplied by your code. It may summarize or rephrase the
retrieved content. If the context is missing or irrelevant, the model
may provide an unsupported answer unless the prompt clearly instructs it
to stay within the evidence.

For a fair test, use the same question for retrieval and generation,
format the retrieved `page_content` as context, and instruct the model
to say when the context does not contain the answer. Always inspect
retrieved chunks and verify the final response against them.

## Sample PDF notice

The included `ABC_Bank_Customer_Support_FAQ_Sample.pdf` is a **fictional
demonstration document**. Its bank contact details, addresses, fees, and
policies are invented sample values. They must not be used for real
banking or customer support.

## Security and good practices

-   Keep API keys in `.env` or a secure secrets manager.
-   Add `.env`, `.venv/`, notebook checkpoints, and local vector-store
    files to `.gitignore`.
-   Do not commit API keys, credentials, or sensitive customer data.
-   Inspect retrieved text before trusting an answer.
-   Treat generated responses as unverified until checked against the
    source.
-   Preserve source metadata, such as page numbers, where possible for
    traceability.

## Learning notes

See
[`RAG_PDF_Semantic_Search_vs_LLM_Guide.md`](RAG_PDF_Semantic_Search_vs_LLM_Guide.md)
for a detailed explanation of the notebook code, semantic search,
context passing, and grounded generation.

