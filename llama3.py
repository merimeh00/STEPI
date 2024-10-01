import transformers
import torch
import gradio as gr
import time
import threading

# Modelo para chatear
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model = model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:0",
)

# Modelo para extraer tripletas
triplet_model_id = "mariavilla/gemma2"

triplet_pipeline = transformers.pipeline(
    "text-generation",
    model=triplet_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:1"
)

def extract_triplets(sentence):
    messages = [ 
        {
        
            "role":"user",
            "content": f"<bos>Given the following sentence: {sentence}, extract the Head, Tail and Label. Where the Head typically refers to the person speaking or the main action or topic being discussed. The Tail usually describes something related to the head, such as their interests, activities, or characteristics. The Label categorizes the sentence into various themes or topics, providing insight into the content of the conversation."
        
        }
    ]
    
    formatted_messages = (
        "<start_of_turn>user\n"
        f"{messages[0]['content']}<end_of_turn>\n"
        "<start_of_turn>model\n"   
    )
    
    triplet_response = triplet_pipeline(formatted_messages, 
                                        max_length=200,
                                        temperature=0.5,
                                        do_sample=True,
                                        top_p=0.7,
                                        eos_token_id=triplet_pipeline.tokenizer.eos_token_id,
                                        pad_token_id=triplet_pipeline.tokenizer.pad_token_id,
                                        bos_token_id=triplet_pipeline.tokenizer.bos_token_id,
                                        #bos_token_id="<bos>",
                                        truncation=True)
    print(triplet_response)
    
    generated_text = triplet_response[0]['generated_text'][len(formatted_messages):]
    
    formatted_text = generated_text.replace('{', '').replace('}', '').replace('Head :', 'Head:').replace('Tail :', 'Tail:').replace('Label :', 'Label:')
    
    return formatted_text

def chat_function(message, history, system_prompt, max_new_tokens, temp):
    messages = [ 
        {
        
            "role":"system",
            "content": system_prompt
        
        },
        {
            "role":"user",
            "content":message
        }
    ]

    formatted_messages = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{messages[0]['content']}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{messages[1]['content']}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    response = pipeline(formatted_messages, 
                    max_length=max_new_tokens,
                    temperature = temp + 0.1,
                    do_sample = True,
                    top_p = 0.9,
                    eos_token_id=pipeline.tokenizer.eos_token_id,
                    pad_token_id=pipeline.tokenizer.pad_token_id,
                    stop_sequence="<|eot_id|>")
    
    chatbot_response = response[0]["generated_text"][len(formatted_messages):]
    print(response)
    return chatbot_response

## FOR TESTING ##  

# def test_triplet_extraction():
#     test_sentence = "John is going to the market to buy some apples."
#     triplets = extract_triplets(test_sentence)
#     print(f"Extracted Triplets: {triplets}")

# test_triplet_extraction()