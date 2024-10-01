import gradio as gr
from llama3 import chat_function, extract_triplets
import time

# Definir la variable global para almacenar el historial de triplets
triplet_history = []

# Parámetros por defecto
DEFAULT_SYSTEM_PROMPT = "You are a conversational AI interested in talking about the users hobbies"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMP = 0.7

def clear_triplet_history():
    global triplet_history
    triplet_history = []
    return ""

def handle_send(message, history, triplet_history_state):
    global triplet_history
    
    # Valores por defecto
    system_prompt = DEFAULT_SYSTEM_PROMPT
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    temp = DEFAULT_TEMP
    
    # Mostrar el mensaje del usuario inmediatamente en el chatbox
    new_history = history + [(message, "")]
    yield gr.update(value=new_history, visible=True), gr.update(visible=True), gr.update(value=""), gr.update(visible=False), gr.update(visible=True)
    
    # Mostrar puntos suspensivos mientras el modelo piensa
    for _ in range(10):
        new_history[-1] = (message, "...")
        yield gr.update(value=new_history, visible=True), gr.update(visible=True), gr.update(value=""), gr.update(visible=False), gr.update(visible=True)
        time.sleep(0.3)
    
    # Generar la respuesta del chatbot y las tripletas
    chatbot_response = chat_function(message, history, system_prompt, max_new_tokens, temp)
    triplet = extract_triplets(message)
    
    # Actualizar el historial de triplets
    triplet_history.append(triplet)
    updated_triplet_history = "\n".join(f"Turn {i * 2 + 1}: {triplet}" for i, triplet in enumerate(triplet_history))
    
    # Mostrar la respuesta del chatbot de forma gradual
    for i in range(1, len(chatbot_response) + 1):
        partial_response = chatbot_response[:i]
        new_history[-1] = (message, partial_response)
        yield gr.update(value=new_history, visible=True), gr.update(value=updated_triplet_history, visible=True), gr.update(value=""), gr.update(visible=False), gr.update(visible=True)
        time.sleep(0.03)
    
    # Actualizar la interfaz final
    yield gr.update(value=new_history, visible=True), gr.update(value=updated_triplet_history, visible=True), gr.update(value=""), gr.update(visible=False), gr.update(visible=True)

css = """
.triplets_prompt span {font-size: 19px !important}
.triplets_output span {font-size: 19px !important}
.zoom-out img {transforms: scale(0.9);}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            image = gr.Image("fondo_prueba4.png", label= "Welcome", height=622, width="auto", elem_classes="zoom-out", visible=True)  # Imagen visible por defecto
            chatbox = gr.Chatbot(height=622, visible=False)  # Chatbox oculto por defecto
            with gr.Row():
                with gr.Column(scale=9):
                    message_input = gr.Textbox(placeholder="Enter a message here", container=False, scale=7)
                send_button = gr.Button("⏎", elem_id="send-button")
        with gr.Column(scale=1):
            triplets_prompt = gr.Textbox("Given the following sentence, extract the Head, Tail and Label."
                                         "Where the Head typically refers to the person speaking or the main action or topic being discussed. "
                                         "The Tail usually describes something related to the head, such as their interests, activities, or characteristics. "
                                         "The Label categorizes the sentence into various themes or topics, providing insight into the content of the conversation.", 
                                         label="Prompt",
                                         elem_classes="triplets_prompt")
            triplets_output = gr.Textbox(label="Extracted Triplets", lines=15, interactive=False, elem_classes="triplets_output")
            clear_button = gr.Button("Clear History") 
            
        # Configura el evento de envío del mensaje al presionar Enter
        message_input.submit(handle_send, 
                             inputs=[message_input, chatbox, gr.State(triplet_history)], 
                             outputs=[chatbox, triplets_output, message_input, image, chatbox])
        
        # Configura el evento de clic del botón para llamar a ambas funciones
        send_button.click(handle_send,
                          inputs=[message_input, chatbox, gr.State(triplet_history)],
                          outputs=[chatbox, triplets_output, message_input, image, chatbox])
        
        # Configura el botón de limpiar historial
        clear_button.click(clear_triplet_history, None, triplets_output)    
        
demo.launch()




