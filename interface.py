import RAG
import gradio as gr
import os

def respond(question):
    return  str(RAG.decompose_question(question))
   
def respond_1(question):
    return RAG.respond_3(question) # 3 chunks at a time and return the answer where the context is the first answer

def respond_2(question):
    text = ""
    L = RAG.get_near_chunks(question)
    for i in range(len(L)):
        text += f" {i+1}) {L[i]} \n"
    return text

def upload(doc):
    if doc is None:
        return "No document uploaded!"
    
    filenames = [os.path.join(doc[i]) for i in range(len(doc))] 

    RAG.prepare_chunks(filenames)
    

    return "Upload successfully!"


with gr.Blocks() as block:
    gr.Markdown("## RAG")
    with gr.Tab("Chat") as respond_tab:
        question = gr.Textbox(label="Question",placeholder="Type your question here")
        response = gr.Textbox(label="Reponse")
        submit = gr.Button("Soumettre")
    with gr.Tab("Upload") as upload_tab:
        doc = gr.File(label="Upload le document",file_count='multiple')
        submit_doc = gr.Button("Soumettre")
        submit_statu = gr.Textbox(placeholder = "Upload un document pour commencer",label="Upload")
    
    submit.click(respond_1, inputs=question, outputs=response)
    submit_doc.click(upload, inputs=doc, outputs=submit_statu)

block.launch()
