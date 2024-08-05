import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

import configparser
config = configparser.ConfigParser()
config.read("config.ini")
openslidebin = fr"{config['prod']['openslidebin']}"

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(openslidebin):
        import openslide
else:
    import openslide
    
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
from utils import Normalizer, make_tfs, CustomDataset, create_model
from tqdm import tqdm
import gc
import re

    
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def make_predictions_per_batch(model: torch.nn.Module, data: torch.utils.data.DataLoader,
                               device: torch.device):
    pred_probs = []
    fns = []
    model.eval()
    with torch.inference_mode():  # disable for grad cam
        for batch in tqdm(data):
            # Prepare sample
            sample = batch['image'].to(device)  # send batch to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)['out']  # ['out'] # 0 is the threshold
            probs = torch.sigmoid(pred_logit).to(torch.float16)

            # Append to output lists
            pred_probs.append(probs)
            fns.append(batch['image_name'])

        # flatten filenames
        fns = [item for sublist in fns for item in sublist]

        # Stack the pred_probs to turn list into a tensor
        return torch.cat(pred_probs, dim=0), fns


stain_normalizer = Normalizer(name='macenko')


class StainNormalize(object):
    """Stain Normalize"""

    def __call__(self, sample):
        data = np.array(sample)
        data = (np.transpose(data, axes=(1, 2, 0)) * 255.0).astype(np.uint8)
        data = stain_normalizer.norm(data)

        return data


def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--cluster_type', type=str, required=True, help='Cluster Type')
    parser.add_argument('--partial_type', type=str, required=True, help='Partial Type')
    parser.add_argument('--fold', type=str, required=True, help='Fold')
    parser.add_argument('--input_directory', type=str, required=True, help='Input Directory')
    parser.add_argument('--output_directory', type=str, help='Output Directory', default='./outputs')
    
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"GPU?: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print("")

    # MANUAL PARAMETERS
    BATCH_SIZE = 8

    print(f"Now predicting:\n      Partial Type: {args.partial_type}\n      Cluster Type: {args.cluster_type}\n      Fold: {args.fold}\n")
    print(f"Input directory: {args.input_directory}")
    print(f"Output directory: {args.output_directory}")
    print("")

    # Set model path
    MODEL_PATH = f"models/{args.partial_type}_{args.cluster_type}/fold_{args.fold}/model.pth"

    # Load base model and our weights
    loaded_model = create_model()  # pretrained from pytorch
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH))

    # Freeze
    for param in loaded_model.parameters():
        param.requires_grad = False
        
    # Send model to gpu
    loaded_model.to(device)

    print(f"Appending model information to output directory...")
    new_output_dir = f"{args.output_directory}/{args.partial_type}_{args.cluster_type}_fold_{args.fold}"
    os.makedirs(new_output_dir, exist_ok=True)
    print(f"Created directory {new_output_dir}\n")

    # Iterate through extracted patches, create dataset for each slide on the fly
    image_dirs = natural_sort(os.listdir(args.input_directory))

    for img_dir in image_dirs:
        if img_dir not in ["55 (JAK)"]:
            continue
        
        print(f"{img_dir}")
        
        # Make output sub-directories
        mask_dir = os.path.join(new_output_dir, img_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)

        # Assemble Dataset
        slide_path = os.path.join(args.input_directory, img_dir)
        list_of_images = natural_sort([os.path.join(slide_path, x) for x in os.listdir(slide_path)])

        geometric_augs = [
        ]

        color_augs = [
            StainNormalize(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        ds = CustomDataset(list_of_images, None,
                            transform=make_tfs(geometric_augs + color_augs),
                            mask_transform=None)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                                    pin_memory=True)


        pr, fns = make_predictions_per_batch(loaded_model, dataloader, device)

        print(f"Saving predictions...")
        # Iterate through predictions and filenames and save
        for p, f in tqdm(zip(pr, fns)):
            fn = os.path.splitext(os.path.basename(f))[0]
            output_path_mask = os.path.join(mask_dir, fn)
            p_numpy = p.squeeze(0).cpu().detach().numpy()
            final_image = ((p_numpy > 0.5) * 255).astype(np.uint8)
            cv2.imwrite(f"{output_path_mask}.png", final_image)
        print("")
    gc.collect()


if __name__ == '__main__':
    main()