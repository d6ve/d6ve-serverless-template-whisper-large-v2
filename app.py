import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    #medium, large-v1, large-v2
    model_name = "large-v2"
    model = whisper.load_model(model_name)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    inputLanguage = model_inputs.get('language', None)
    if inputLanguage == None:
        return {'message': "No language provided"}
    
    inputTask = model_inputs.get('task', None)
    if inputTask == None:
        return {'message': "No task provided"}
        
    args = dict(
        language = (None if inputLanguage.lower() == "auto" else inputLanguage),
        task = inputTask
    )
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    result = model.transcribe("input.mp3",**args)
    os.remove("input.mp3")

    # Return the results as a dictionary
    return result
