from fastai.vision.all import *
import gradio as gr
import os


def label_func(f): 
    return f[0].isupper()

learn = load_learner('model_Breed_RES34_460_32_epoch4.pkl')
learn1 = load_learner('modell.pkl')


categories = learn.dls.vocab
categories1 = ['Dog', 'Cat']


def predict(img):
    img = PILImage.create(img)
    probs1 = learn.predict(img)[2]
    probs2 = learn1.predict(img)[2]
    
   
    top_breeds1 = sorted(zip(categories, probs1), key=lambda x: x[1], reverse=True)[:3]
    top_breeds2 = sorted(zip(categories1, probs2), key=lambda x: x[1], reverse=True)[:3]

  
    result = {breed: float(prob) for breed, prob in top_breeds1}
    result1 = {breed: float(prob) for breed, prob in top_breeds2}

 
    result.update(result1)
    return result

image = gr.Image()
label = gr.Label()


script_dir = os.path.dirname(os.path.abspath(__file__))
example = os.path.join(script_dir, "Etku.jpg")


if not os.path.exists(example):
    print("Error: Image file 'Etku.jpg' not found in the same directory.")
    exit(1)


interface = gr.Interface(
    fn=predict, 
    inputs =image, 
    outputs = label, 
    examples = [[example]],
    title = "Pet Breed Classifier",
    description = "This model predicts the breed of the",
    theme='soft',
    allow_flagging= 'never',
    live = true,
    
   )


interface.launch(share=True)