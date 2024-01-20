import RAG
import gradio as gr
import os

def respond(question):
    return RAG.respond(question)
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

    RAG.prepare_chunks(filename=filenames[0])
    
    for filename in filenames[1:]:
        RAG.prepare_chunks(filename=filename,clear=False)

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
    
    submit.click(respond_2, inputs=question, outputs=response)
    submit_doc.click(upload, inputs=doc, outputs=submit_statu)

block.launch()






