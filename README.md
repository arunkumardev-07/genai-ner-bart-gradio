## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
To develop a Named Entity Recognition (NER) prototype that identifies and categorizes named entities such as persons, locations, and organizations from text input using a fine-tuned transformer-based model (dslim/bert-base-NER). The prototype must be deployed with an interactive user interface using the Gradio framework.

### DESIGN STEPS:

#### STEP 1: Load Required Libraries and Model
Imported transformers, gradio, and dotenv libraries and initialized the dslim/bert-base-NER model pipeline with aggregation_strategy="simple" for grouped entity outputs.

#### STEP 2: Prepare NER Inference Function
Created a function to process user input text and extract entity spans (substring + label) suitable for visual rendering.

#### STEP 3: Build Gradio Interface
Used gr.Interface with gr.Textbox as input and gr.HighlightedText as output to visualize entities directly in the input sentence.

### STEP 4: Add Example Inputs and Descriptions
Enhanced the interface with predefined examples and titles to guide the user.

### STEP 5: Deploy and Test the Application
Launched the app locally (or via public link using share=True) and validated with various test inputs for people, locations, and organizations.

### PROGRAM:
```
import os
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from transformers import pipeline

# Load .env
_ = load_dotenv(find_dotenv())

# Load the NER model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Correct function for gr.HighlightedText
def ner_highlight(text):
    ner_results = ner_pipeline(text)
    spans = []
    for ent in ner_results:
        entity_text = text[ent["start"]:ent["end"]]
        label = ent["entity_group"]
        spans.append((entity_text, label))  # string span + entity label
    return spans

# Gradio app
gr.close_all()
demo = gr.Interface(
    fn=ner_highlight,
    inputs=gr.Textbox(label="Text to find entities", lines=4, placeholder="Type a sentence..."),
    outputs=gr.HighlightedText(label="Text with entities"),
    title="NER with dslim/bert-base-NER",
    description="Find entities like persons, organizations, and locations using the `dslim/bert-base-NER` model.",
    examples=[
        "My name is Arunkumar, I'm building Blockchain and I live in Chennai",
        "My name is Poli, I live in Vienna and work at HuggingFace",
        "Narendra Modi was born in Gujarat and became Prime Minister of the India.",
        "Elon Musk founded SpaceX in California."
    ],
    allow_flagging="never"
)

demo.launch(share=True, server_port=int(os.environ.get("PORT3", 7860)))

```

### OUTPUT:
![WhatsApp Image 2025-05-17 at 11 57 40_06a69f50](https://github.com/user-attachments/assets/77f22b12-da93-44ed-9d3e-db4f4695c9b8)


### RESULT:
The application successfully identifies and highlights entities such as names, cities, countries, and companies from user input.

Output is rendered with entity labels (e.g., PER, LOC, ORG) directly in the sentence using colored highlights.

