# 
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
import os

# import OpenAI API key from secret.txt
with open("secret.txt", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

def create_index(filenames: list[str]):
    # TODO: make a list of files to index
    llm = OpenAI(openai_api_key="OPENAI_API_KEY")
    loaders = [UnstructuredPDFLoader(fname) for fname in filenames]
    index = VectorstoreIndexCreator().from_loaders(loaders)

    return index 


# TODO: select files
# TODO: tool use

# persist
# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html

# enter an endless loop with user input
def main():
    print("Welcome to FIBSEM Chat! Enter your question, or type 'quit' to exit.")

    # load user manual, index
    filenames = ["documents/OnlineManualHeliosHydra.pdf"]
    # filenames = ["documents/autoliftout.pdf", "documents/supplementary.pdf"]

    print(f"Creating index from {filenames}.")
    index =  create_index(filenames=filenames)
    print(f"Index created from {filenames}.\n")

    while True:

        q = input("Enter a question: ")

        if q in ["quit", "exit"]:
            print("\nGoodbye!")
            break
        
        print("Thinking...")

        # run query, show result
        ret = index.query(q)   

        print("\n\nResponse: " + ret)
        print("-"*50)


if __name__ == "__main__":
    main()
