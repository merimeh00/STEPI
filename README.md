# STEPI.-System-for-Triplet-Extraction-of-Personal-Information-in-Dialogue-Systems
Personalization is a crucial step in dialogue systems, as it enhances the user experience by providing customized responses. In this paper, we introduce an automatic system for extracting personal information provided during the user interaction. The extracted information is formatted as triplets, consisting of a Head, Tail and Label. We fine-tuned and evaluated five Small Language Models (SLMs): Microsoft Phi 1.5, Phi 2, Phi 3, Gemma 2, and Llama 3-8B. These models were fine-tuned on the PersonaExt dataset, created by the authors of the Persona Attribute Extractor Detection (PAED). The final chatbot is developed using two models, the LLaMA 3 (8B) without fine-tuning in order to generate the responses, and Gemma 2 fine-tuned with the 80% of the PersonaExt database. 

The chatbot was developed using Gradio.

