# Understanding RAG with a PDF: Semantic Search vs. LLM Responses

## Purpose of this guide

This guide explains the code in `RAG_PDF.ipynb` step by step and focuses
on a key question:

**Why does the output of semantic search look different from the
response generated after passing the retrieved context to an LLM?**

The notebook uses a two-page sample PDF,
`ABC_Bank_Customer_Support_FAQ_Sample.pdf`, with LangChain, OpenAI
embeddings, Chroma, and an OpenAI chat model.

> **Important:** The PDF labels its bank contacts, branch details, fees,
> and policies as fictional demonstration values. Use them only to
> understand and test the RAG pipeline---not as real banking
> information.

------------------------------------------------------------------------

## 1. The RAG workflow

RAG stands for **Retrieval-Augmented Generation**. It combines finding
relevant information from a document with generating a response from
that information.

``` text
Sample PDF
   |
   v
PyPDFLoader extracts text
   |
   v
Documents + metadata
   |
   v
Text splitter creates chunks
   |
   v
Embedding model converts chunks to vectors
   |
   v
Chroma stores text, vectors, and metadata
   |
   v
User question -> semantic search
   |
   v
Retrieved document chunks
   |
   v
Question + retrieved context -> LLM
   |
   v
Generated natural-language answer
```

The key distinction is:

-   **Semantic search retrieves relevant-looking source text.**
-   **The LLM generates a response using the prompt and context it
    receives.**

They are related steps, but they do different jobs.

------------------------------------------------------------------------

## 2. Load environment variables

Notebook code:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key is configured.")
else:
    raise ValueError("OPENAI_API_KEY is missing from .env")
```

### What each part does

-   `os` provides access to environment variables.
-   `load_dotenv()` reads variables from a local `.env` file and makes
    them available to Python.
-   `os.getenv("OPENAI_API_KEY")` reads the configured key. It returns
    `None` if the variable is missing.
-   The `if/else` check stops the notebook early with a clear error if
    the key is missing.

This confirms that a value is configured; it does not prove that the key
is valid or has API access.

**GitHub safety:** keep `.env` out of the repository by adding it to
`.gitignore`. Never commit your actual API key or place it directly in a
notebook.

------------------------------------------------------------------------

## 3. Initialize the LLM

Notebook code:

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0
)
```

-   `ChatOpenAI` provides a LangChain interface to an OpenAI chat model.
-   `model` chooses which model will generate the response.
-   `temperature=0` requests less variation in the model's output. It
    does not guarantee factual correctness or prevent unsupported
    answers.
-   `llm` is the configured model object. It does not automatically read
    the PDF or query Chroma.

The LLM will receive information only when your code sends it in a
prompt or through a chain/tool configured to provide that information.

------------------------------------------------------------------------

## 4. Extract text from the PDF

Notebook code:

``` python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "./Docs/ABC_Bank_Customer_Support_FAQ_Sample.pdf"
)

docs = loader.load()
docs
```

### What happens?

1.  `PyPDFLoader` opens the PDF at the supplied path.
2.  `loader.load()` extracts text and returns a list of LangChain
    `Document` objects.
3.  With this loader, a PDF is generally represented as one document per
    page.

A `Document` commonly has two important fields:

-   `page_content`: the extracted text from that page.
-   `metadata`: details about the source, often including the file path
    and page number.

For example, the sample PDF's second page includes fictional branch
addresses for Bengaluru, Hyderabad, Chennai, and Mumbai. That text can
be retrieved later if it was extracted correctly and included in the
indexed chunks.

**Why this matters:** the vector store searches the extracted text, not
the visual appearance of the PDF. If a PDF contains scanned pages,
complex tables, or text-extraction problems, the text available to
retrieval may be incomplete.

------------------------------------------------------------------------

## 5. Add metadata to each document

Notebook code:

``` python
for i in docs:
    i.metadata = {
        "source": "ABC_Bank_Customer_Support_FAQ_Sample.pdf",
        "developer": "Chethan Prabhas",
    }
```

Metadata is descriptive information associated with a document or chunk.
It can help identify a source, filter retrieval, and trace where a
retrieved passage came from.

### A detail to understand

The assignment `i.metadata = {...}` replaces the existing metadata
dictionary. This can remove metadata created by `PyPDFLoader`, such as
the page number.

A version that preserves existing metadata is:

``` python
for doc in docs:
    doc.metadata.update({
        "source": "ABC_Bank_Customer_Support_FAQ_Sample.pdf",
        "developer": "Chethan Prabhas",
    })
```

`update()` adds or changes the specified keys while retaining other
metadata keys.

For a public GitHub repository, you may prefer a neutral project/author
label and avoid personal details unless you intentionally want them
included.

------------------------------------------------------------------------

## 6. Split the PDF into chunks

Notebook code:

``` python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
chunks
```

### Why chunk the text?

Retrieving an entire long document is often less useful than retrieving
the sections that relate to the user's question. Chunking breaks each
page/document into smaller text units that can be indexed and retrieved
separately.

-   `chunk_size=1000`: targets chunks of around 1,000 **characters**,
    not 1,000 words or tokens.
-   `chunk_overlap=100`: neighboring chunks can share around 100
    characters. Overlap helps preserve meaning when a passage crosses a
    chunk boundary.
-   `split_documents(docs)`: splits each input document and carries its
    metadata into the resulting chunks.

### Does a two-page PDF need chunking?

Not automatically. Page count alone does not determine the number of
chunks. The amount and structure of extracted text and the splitting
settings determine the result. For a short PDF, compare chunking with
using each page as a retrieval unit; choose the approach that retrieves
useful evidence for your questions.

To inspect the result:

``` python
print("Number of PDF page documents:", len(docs))
print("Number of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"Chunk {index}: {len(chunk.page_content)} characters")
    print("Metadata:", chunk.metadata)
```

------------------------------------------------------------------------

## 7. Create embeddings

Notebook code:

``` python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

An **embedding** is a numerical vector representing text. An embedding
model maps text into a vector space in which texts with related meanings
can often be found near one another.

For example, the question:

> "What is the phone number of ABC Bank in Bangalore?"

may be related to a chunk that contains the Bengaluru branch phone
number, even if the chunk uses the spelling "Bengaluru."

The code configures the embedding model; it does not itself generate a
conversational answer. The model is used when the chunks and later the
query are embedded for similarity search.

------------------------------------------------------------------------

## 8. Create the Chroma vector store

Notebook code:

``` python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)
```

At a high level, this step:

1.  Takes the chunks.
2.  Uses the embedding model to create a vector for each chunk.
3.  Stores the vectors with the associated text and metadata in Chroma.

Later, Chroma can compare a question's embedding with the stored chunk
embeddings and return similar chunks.

No persistent directory is specified in this code, so the store is
typically created for the current runtime/session rather than saved as a
durable local database. If you want persistence, configure a persistence
directory and follow the behavior supported by your installed
Chroma/LangChain versions.

------------------------------------------------------------------------

## 9. Perform semantic search

Notebook code:

``` python
results = vectorstore.similarity_search(
    "what is the phone number of ABC Bank in Bangalore?",
    k=2
)

for index, doc in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(doc.page_content)
    print(doc.metadata)
```

### What `similarity_search()` does

-   Takes the question as a search query.
-   Uses the configured embedding function to represent the query as a
    vector.
-   Compares it with the stored chunk vectors.
-   Returns up to `k=2` of the most similar chunks as `Document`
    objects.

The loop prints the retrieved text and metadata so you can inspect the
evidence that the retriever found.

### What semantic search does not do

It does not automatically write a polished answer, combine the passages
into a summary, or guarantee that the top results fully answer the
question. It returns chunks based on similarity.

For this sample PDF, the second page includes a Bengaluru demo branch
and a phone number. The retrieved chunk may include that information,
depending on how the text was split and ranked.

**Good practice:** inspect the retrieved chunks before judging the final
answer. If the right branch number is not present in the retrieved
context, the issue may be retrieval, chunking, or extraction---not
necessarily the LLM.

------------------------------------------------------------------------

## 10. Pass retrieved context to the LLM

Notebook code:

``` python
context = vectorstore.similarity_search(
    "what is the phone number of ABC Bank in Bangalore?",
    k=2
)

response = llm.invoke(
    f"what is the phone number of ABC Bank in Bangalore? "
    f"You can answer using the following context: {context}"
)

print(response.content)
```

### Line by line

1.  `similarity_search(..., k=2)` retrieves two chunks again. This means
    `context` is a list of `Document` objects, not a plain string of
    only the page text.
2.  The f-string inserts the string representation of those documents
    into the prompt.
3.  `llm.invoke(...)` sends the prompt to the chat model.
4.  `response.content` prints the text generated by the model.

### Why can this output differ from semantic search?

  -----------------------------------------------------------------------
  Semantic search output              LLM response
  ----------------------------------- -----------------------------------
  A list of source `Document` objects Generated natural-language text

  Shows the retrieved passages and    May summarize or extract a detail
  metadata                            from the passages

  Retrieval is based on semantic      Generation is based on the full
  similarity                          prompt, context, and the model's
                                      learned patterns

  Does not itself answer in           Can produce a direct answer,
  conversational form                 explanation, or---if poorly
                                      grounded---unsupported details
  -----------------------------------------------------------------------

The LLM is **not simply printing the retrieved chunks**. It reads the
prompt and generates a new response. It may shorten or rephrase the
retrieved information.

### One important weakness in the current prompt

The instruction says:

> "You can answer using the following context"

The phrase **"can answer using"** makes the context optional. It does
not clearly tell the model that it must base its answer only on the
retrieved text. If the context is irrelevant or incomplete, the model
may use general knowledge or guess. That is one reason RAG applications
need clear grounding instructions and retrieval evaluation.

Also, the list of `Document` objects is being interpolated directly. It
is clearer to explicitly format the `page_content` and, if needed, the
metadata before passing the context to the LLM.

------------------------------------------------------------------------

## 11. The final cell intentionally asks a different question

The notebook's final code cell uses the same retrieved context but asks:

``` python
context = vectorstore.similarity_search(
    "what is the phone number of ABC Bank in Bangalore?",
    k=2
)

response = llm.invoke(
    f"what is the phone number of ICICIC Bank in Bangalore? "
    f"You can answer using the following context: {context}"
)

print(response.content)
```

**Notice the mismatch:**

-   Retrieval query: "What is the phone number of **ABC Bank** in
    Bangalore?"
-   LLM question: "What is the phone number of **ICICIC Bank** in
    Bangalore?"

The retriever searches the ABC Bank sample PDF, but the LLM is asked for
ICICIC Bank's number. The retrieved text cannot reliably answer that
question because the PDF is about fictional ABC Bank demo information,
not ICICIC Bank.

This is a particularly useful demonstration: **retrieval and generation
can disagree because the retrieval query and the final question do not
match.** The LLM might say that the context does not provide the ICICIC
number, or it might produce an unsupported answer. You should not treat
a plausible response as proof that the retrieved PDF supports it.

For a fair comparison, keep the query and the LLM question identical.

------------------------------------------------------------------------

## 12. Improved version: format context and require grounding

Here is a clearer pattern to use after building `vectorstore`:

``` python
question = "What is the phone number of ABC Bank in Bengaluru?"

retrieved_docs = vectorstore.similarity_search(
    question,
    k=2
)

# Convert the retrieved Document objects into plain text.
context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

prompt = f"""
Answer the question using only the context below.

If the context does not contain the answer, say:
"I couldn't find that information in the provided document."

Treat the context as the source of truth. Do not invent
phone numbers or use outside knowledge to fill gaps.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)
print(response.content)
```

### Why this is an improvement

-   **One `question` variable** helps keep the retrieval query and
    generation question consistent.
-   `"\n\n".join(...)` extracts and separates the actual text from each
    retrieved `Document`.
-   The prompt explicitly instructs the LLM to use only the supplied
    context.
-   A fallback response is specified for missing evidence.
-   It warns against inventing phone numbers.

These instructions improve the chance of a grounded answer, but prompts
are not absolute guarantees. The answer should still be checked against
the retrieved text.

To see what evidence the model received, print the chunks as well:

``` python
for index, doc in enumerate(retrieved_docs, start=1):
    print(f"\n--- Retrieved chunk {index} ---")
    print("Metadata:", doc.metadata)
    print(doc.page_content)
```

For traceability, preserving page-number metadata can help identify the
source page of the answer.

------------------------------------------------------------------------

## 13. How to debug a wrong or unexpected answer

When the final answer seems wrong, check these items in order:

1.  **Check the question:** Is the retrieval query the same as the LLM
    question? The last notebook cell currently uses ABC Bank for
    retrieval but ICICIC Bank for generation.
2.  **Inspect the retrieved chunks:** Does the text actually contain the
    requested fact?
3.  **Check extraction:** Was the relevant PDF text extracted correctly?
4.  **Check chunking:** Was the relevant information split across chunks
    or mixed with unrelated text?
5.  **Check retrieval settings:** Is `k=2` enough? Try a different `k`
    and compare the retrieved evidence.
6.  **Check prompt wording:** Does the prompt require answers to be
    based only on the context, or merely say the context may be used?
7.  **Check the generated answer:** Is every specific detail supported
    by the retrieved passages?

Changing `k` can help retrieve more evidence, but increasing it blindly
may add irrelevant text. Evaluate retrieval quality with questions you
expect the document to answer.

------------------------------------------------------------------------

## 14. Common misunderstandings

### "Semantic search answers my question."

It retrieves text that is semantically similar. You must check whether
the retrieved text contains the answer.

### "The LLM automatically has access to Chroma."

No. In this notebook, your Python code retrieves the documents and
explicitly passes their content into the prompt.

### "Passing context forces the model to follow it."

No. Context and prompt instructions influence the response but do not
guarantee perfect grounding.

### "The LLM response must match the retrieved text word for word."

No. The model generates a new response and may paraphrase or summarize.
It should preserve the meaning and avoid unsupported details.

### "A confident answer means it came from the PDF."

No. A fluent response can still be unsupported. Compare it with the
actual retrieved chunks.

### "Two PDF pages mean two chunks."

Not necessarily. The number of chunks depends on the extracted text,
splitting configuration, and document boundaries.

------------------------------------------------------------------------

## 15. Summary

-   **PDF loader:** extracts text and metadata from the source.
-   **Chunking:** breaks text into smaller retrieval units.
-   **Embeddings:** represent text as vectors for similarity comparison.
-   **Chroma:** stores the vectors and their associated chunks.
-   **Semantic search:** retrieves chunks similar to a query.
-   **LLM generation:** produces a natural-language response from the
    prompt and supplied context.
-   **Grounded prompting:** tells the LLM to answer only from retrieved
    evidence and to admit when evidence is missing.

The central lesson is:

> **Retrieval finds potentially useful evidence. Generation turns the
> evidence into an answer. A good RAG system needs both relevant
> retrieval and faithful generation.**

For this notebook in particular, the most important issue to notice is
the final cell's mismatch: it retrieves using an ABC Bank question but
asks the LLM about ICICIC Bank. Keep the question consistent, format the
retrieved text explicitly, and inspect the evidence before trusting the
answer.
