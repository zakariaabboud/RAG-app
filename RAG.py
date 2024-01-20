from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flashrank import Ranker, RerankRequest
import pypdfium2 as pypdfium
import openai
import numpy as np
import json

openai.api_key = "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"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
ranker = Ranker(model_name="ms-marco-MultiBERT-L-12")

def read_file(filename):
    """Read the pdf file and return the text."""
    pdf = pypdfium.PdfDocument(filename)
    text = ""
    for page in pdf:
        text_page = page.get_textpage().get_text_range().replace('\r\n',' ')
        text += text_page
    return text


def chunk(text):
    """Chunk the text into sentences and return a list of sentences."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators= ["\n\n","\n","."," ",""])
    return splitter.split_text(text)
        

def get_embeddings(model,sentences):
    """Get the embeddings of the sentences."""
    embeddings = model.encode(sentences)
    return embeddings

def prepare_chunks(filename,clear = True):
    """Read the file and prepare the chunks."""

    file_dir = "chunks.json"

    text = read_file(filename)
    sentences = chunk(text)
    embeddings = get_embeddings(embedding_model,sentences)

    if clear:
        data = {"embeddings":{},"sentences":[]}
    else:
        with open(file_dir,'r') as json_file:
            data = json.load(json_file)

    for i in range(len(sentences)):
        data["embeddings"][sentences[i]] = embeddings[i].tolist()
        data["sentences"].append(sentences[i])

    with open(file_dir, 'w') as outfile:
        json.dump(data, outfile)
    

def get_similarity(embeddings_1,embeddings_2):
    """Get the similarity between two embeddings."""
    return cosine_similarity(embeddings_1,embeddings_2)

def rerank(query, chunks, n = 10):
    """Rerank the chunks."""
    passages = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]

    request = RerankRequest(query=query, passages=passages)
    response = ranker.rerank(request)
    return response[:n]

def get_near_chunks(text,n = 30):
    """Get the chunks that are the most similar to the text."""
    file_dir = "chunks.json"

    with open(file_dir) as json_file:
        data = json.load(json_file)

    embeddings_1 = get_embeddings(embedding_model,[text])
    embeddings_2 = np.array(list(data["embeddings"].values()))

    similarity = get_similarity(embeddings_1,embeddings_2)
    # n nearest chunks
    index = np.argsort(similarity[0])[-1:-n-1:-1]

    chunks = data["sentences"]
    near_chunks = []
    for i in index:
        # get the chunks with 3 sentences before and after
        #near_chunks.append('.'.join(chunks[i-3:i+4]))
        near_chunks.append(chunks[i])

    near_passages = rerank(text,near_chunks)
    near_chunks = [near_passages[i]["text"] for i in range(len(near_passages))]

    return near_chunks

def get_completion(messages, model="gpt-3.5-turbo", temperature=1):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0]['message']['content']

def respond(question):
    """Get the answer to the question."""
    
    context = get_near_chunks(question)

    prompt = "Répond à la question en utilisant le contexte suivant. La réponsedoit être avec tes propres mot.\n + Context :\n"
    for i in context:
        prompt += "- " + i + "\n"
    prompt += "\nQuestion : " + question + "\nRéponse :"
    message = [{"role": "user", "content": prompt}]
    
    gpt_respond = get_completion(message)

    

    return gpt_respond

    






