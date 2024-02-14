import RAG
import gradio as gr
import os


   
def respond_1(question):
    """Get the answer to the question """
    return RAG.respond_3(question) # 3 chunks at a time and return the answer where the context is the first answer

def respond_2(question):
    """ get the near chunks of the question """
    text = ""
    L = RAG.get_near_chunks(question)
    for i in range(len(L)):
        text += f" {i+1}) {L[i]} \n"
    return text

def upload(doc):
    if doc is None:
        return "Aucun document n'a été chargé"
    
    filenames = [os.path.join(doc[i]) for i in range(len(doc))] 

    RAG.prepare_chunks(filenames)
    

    return "Chargement réussi"


with gr.Blocks() as block:
    gr.Markdown("## RAG")
    with gr.Tab("Chat") as respond_tab:
        question = gr.Textbox(label="Question",placeholder="Écris ta question ici")
        response = gr.Textbox(label="Reponse")
        submit = gr.Button("Soumettre")
    with gr.Tab("Upload") as upload_tab:
        doc = gr.File(label="charger le document PDF",file_count='multiple')
        submit_doc = gr.Button("Soumettre")
        submit_statu = gr.Textbox(placeholder = "Charge un document pour commencer",label="Upload")
    
    submit.click(respond_1, inputs=question, outputs=response)
    submit_doc.click(upload, inputs=doc, outputs=submit_statu)

block.launch(share=True)
