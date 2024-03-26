from fastai.vision.all import *
import gradio as gr
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def main():
    
        path = untar_data(URLs.PETS)
        Path.BASE_PATH = path

        pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75,)
                )
        dls = pets.dataloaders(path/"images", bs=32,num_workers=0) 


        learn = vision_learner(dls, resnet34, metrics=error_rate)
        learn.fine_tune(4)

       
        learn.export('model_Breed_RES34_460_32_epoch4.pkl')

if __name__ == '__main__':
        main()
