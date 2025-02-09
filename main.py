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

OLLAMA_MODEL = "deepseek-r1:8b"
EMBEDDING_DIRECTORY = "./chroma_langchain_db"

app = typer.Typer(no_args_is_help=True)

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
    Pulls a Git repo

    :param name:
    :return:
    """
    git_repo = typer.prompt("Which Git repo would you like to pull?", default="https://github.com/bast/somepackage")
    git_branch = typer.prompt("Which branch would you like to use?", default="master")
    load_repo(git_repo=git_repo, git_branch=git_branch)



def load_repo(git_repo: str, git_branch: str):
    with TemporaryDirectory() as tmpdirname:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
        ) as progress:
            clone_task = progress.add_task(description="Cloning Git repository...", total=1)
            loader = GitLoader(
                clone_url=git_repo,
                repo_path=tmpdirname,
                file_filter=lambda file_path: file_path.endswith(".py"),
                branch=git_branch,
            )
            progress.update(clone_task, completed=True, advance=1)

            split_task = progress.add_task(description="Splitting documents...", total=1)
            python_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=1000, chunk_overlap=0
            )
            documents = loader.load_and_split(text_splitter=python_splitter)
            progress.update(split_task, completed=True, advance=1)

            add_task = progress.add_task(description="Adding documents to vector store...", total=1)
            vector_store.add_documents(documents=documents)
            progress.update(add_task, completed=True, advance=1)

@app.command()
def ask(question: str):
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
    ) as progress:
        clone_task = progress.add_task(description="Asking the LLM...", total=1)
        resp = qa.invoke(question)
        progress.update(clone_task, completed=True, advance=1)
        print(resp.get('result'))


if __name__ == "__main__":
    app()
