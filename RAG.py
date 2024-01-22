from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flashrank import Ranker, RerankRequest
from decouple import config
import concurrent.futures
import pypdfium2 as pypdfium
import openai
import numpy as np
import json


embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
ranker = Ranker(model_name="ms-marco-MultiBERT-L-12")
Client = openai.OpenAI(api_key=config("api"))

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
        chunk_size=300,
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

def rerank(query, chunks, n = 5):
    """Rerank the chunks."""
    passages = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]

    request = RerankRequest(query=query, passages=passages)
    response = ranker.rerank(request)
    return response[:n]

def get_near_chunks(text,n = 6):
    """Get the chunks that are the most similar to the text."""
    
    file_dir = "chunks.json"

    with open(file_dir) as json_file:
        data = json.load(json_file)

    embeddings_1 = get_embeddings(embedding_model,[text])
    embeddings_2 = np.array(list(data["embeddings"].values()))

    similarity = get_similarity(embeddings_1,embeddings_2)
    # 30 nearest chunks
    index = np.argsort(similarity[0])[-1:-30-1:-1]

    chunks_index = { data["sentences"][i]:i for i in index }

    chunks = data["sentences"]
    near_chunks = []

    for i in index:
        near_chunks.append(chunks[i])

    near_passages = rerank(text,near_chunks,n)

    # nearest 5 chunks with 2 chunks before and 2 chunks after
    near_chunks = [ "".join(data["sentences"][ chunks_index[i["text"]] - 2 : chunks_index[i["text"]] + 3 ])
                    for i in near_passages ]
    
    #near_chunks = [ data["sentences"][ chunks_index[i["text"]] ] for i in near_passages ]

    #near_chunks = [ data["sentences"][i] for i in index ]

    return near_chunks

def get_completion(messages, model="gpt-3.5-turbo",temperature=0):
    
    response = Client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
    )

    return response.choices[0].message.content

def decompose_question(question):
    """Decompose the question into sentences."""

    prompt = "Décompose si nécessaire la question suivante en questions pertinentes ( maximum 3 questions et minimum 1 question) : \n"
    prompt += question + "\n"
    prompt += "les questions doivent être séparées par un retour à la ligne. \n"

    messages = [{"role": "user", "content" : prompt}]

    gpt_respond = get_completion(messages)

    questions = gpt_respond.split("\n")

    for i in range(len(questions)):
        questions[i] = questions[i][3:]

    return questions

def get_answer(question,context,temperature=0):
    """Get the answer to the question. The context is a list of sentences."""

    messages = []
    prompt = "Répond à la question en utilisant le contexte suivant :  \n"
    for i in context:
        prompt += i + "\n"
    prompt += "La réponse doit être avec tes propres mot.\n"
    messages.append({"role": "system", "content" : prompt})
    messages.append({"role": "user", "content" : question})
    
    gpt_respond = get_completion(messages,temperature=temperature)
    return gpt_respond




def respond_1(question):
    """Get the answer to the question."""
    
    context = get_near_chunks(question, n = 4)
    
    return get_answer(question,context)

def respond_2(question):
    """Get the answer to the question."""
    
    context = get_near_chunks(question)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        query_1 = executor.submit(get_answer,question,context[:3])
        query_2 = executor.submit(get_answer,question,context[3:])

        answer_1 = query_1.result()
        answer_2 = query_2.result()
    
    context = [answer_1,answer_2]

    return get_answer(question,context)



def respond_3(question):

    questions = decompose_question(question)
    
    dict_answers = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(questions)):
            dict_answers[i] = executor.submit(respond_1,questions[i])
        
        answers = []
        for i in range(len(questions)):
            answers.append(dict_answers[i].result())

    context = answers

    return get_answer(question,context,temperature=0.8)


    






