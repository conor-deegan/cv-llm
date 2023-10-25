# imports
from flask import Flask, jsonify, request
import logging
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

# config
app = Flask(__name__)

openai_api_key = os.environ.get("OPEN_AI_API_KEY")

llm = OpenAI(openai_api_key=openai_api_key)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set up rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "50 per hour", "2 per second"],
)


# training function
def train():
    logger.info("starting training")

    # open cv
    with open("data/cv.txt") as f:
        cv = f.read()

    # create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # split the text into chunks
    texts = text_splitter.create_documents([cv])

    # define the embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # use the text chunks and the embeddings model to fill our vector store
    global db
    db = Chroma.from_documents(texts, embeddings)

    logger.info("finished training")


train()


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "conordeegan.dev"})


@app.route("/healthcheck", methods=["GET"])
def health():
    return jsonify({"message": "conordeegan.dev"})


@app.route("/", methods=["POST"])
def main():
    # get question
    data = request.json
    question = data.get("question", None)
    # check if question is provided
    if question is None:
        return jsonify({"error": "no question provided"}), 400

    logger.info(f"question: {question}")

    # use our vector store to find similar text chunks
    results = db.similarity_search(query=question, n_results=5)

    # define the prompt template
    template = """
    You are a chat bot who loves to help people! Given the following context sections, answer the
    question using only the given context. If you are unsure and the answer is not
    explicitly written in the documentation, say "Sorry, I don't know how to help with that. Only answer with full sentances"

    Context sections:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # fill the prompt template
    prompt_text = prompt.format(context=results, question=question)

    # generate response
    result = llm(prompt_text)

    logger.info(f"answer: {result}")

    # return response
    res = {"answer": result}
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True)
