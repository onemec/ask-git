from tempfile import TemporaryDirectory

import typer
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Constants for the Ollama model and embedding directory
OLLAMA_MODEL = "deepseek-r1:8b"
EMBEDDING_DIRECTORY = "./chroma_langchain_db"

# Initialize Typer app
app = typer.Typer(no_args_is_help=True)

# Initialize the LLM, embeddings, vector store, retriever, and QA chain
llm = ChatOllama(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=EMBEDDING_DIRECTORY,
)
retriever = vector_store.as_retriever()
qa = RetrievalQA.from_llm(
    llm=llm,
    retriever=retriever,
)


@app.command()
def pull():
    """
    Pulls a Git repository and processes its contents.

    Prompts the user for a Git repository URL and branch, then calls the load_repo function.

    :return: None
    """
    git_repo = typer.prompt(
        "Which Git repo would you like to pull?",
        default="https://github.com/bast/somepackage",
    )
    git_branch = typer.prompt("Which branch would you like to use?", default="master")
    load_repo(git_repo=git_repo, git_branch=git_branch)


def load_repo(git_repo: str, git_branch: str):
    """
    Clones a Git repository, splits its documents, and adds them to a vector store.

    Uses a temporary directory to clone the repository, splits the documents using a text splitter,
    and adds the documents to a vector store. Progress is displayed using a spinner.

    :param git_repo: URL of the Git repository to clone
    :param git_branch: Branch of the Git repository to use
    :return: None
    """
    with (
        TemporaryDirectory() as tmpdirname,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress,
    ):
        # Task: Cloning the Git repository
        clone_task = progress.add_task(description="Cloning Git repository...", total=1)
        loader = GitLoader(
            clone_url=git_repo,
            repo_path=tmpdirname,
            file_filter=lambda file_path: file_path.endswith(".py"),
            branch=git_branch,
        )
        progress.update(clone_task, completed=True, advance=1)

        # Task: Splitting the documents
        split_task = progress.add_task(description="Splitting documents...", total=1)
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=0
        )
        documents = loader.load_and_split(text_splitter=python_splitter)
        progress.update(split_task, completed=True, advance=1)

        # Task: Adding documents to the vector store
        add_task = progress.add_task(
            description="Adding documents to vector store...", total=1
        )
        vector_store.add_documents(documents=documents)
        progress.update(add_task, completed=True, advance=1)


@app.command()
def ask():
    """
    Asks a question to the LLM and prints the response.

    Displays a spinner while the LLM processes the question.

    :return: None
    """
    question = typer.prompt(
        "What do you want to ask the LLM about the repo?"
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        # Task: Asking the LLM
        clone_task = progress.add_task(description="Asking the LLM...", total=1)
        resp = qa.invoke(question)
        progress.update(clone_task, completed=True, advance=1)
        print(resp.get("result"))


if __name__ == "__main__":
    app()
