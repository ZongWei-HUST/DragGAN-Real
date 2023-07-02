import os
import pickle
import torch
from configs import paths_config, hyperparameters, global_config
from scripts.run_pti import run_PTI
import dlib
from alignment import align_face
from argparse import ArgumentParser


output_data_path = './image_processed'
global_config.device = 'cuda'


def pre_process_image(raw_image_path):
    """align data."""
    IMAGE_SIZE = 1024
    predictor = dlib.shape_predictor(paths_config.dlib)

    try:
        aligned_image = align_face(filepath=raw_image_path, predictor=predictor, output_size=IMAGE_SIZE)
    except Exception as e:
        print(e)

    img_name = os.path.basename(raw_image_path)
    real_name = img_name.split('.')[0]

    os.makedirs(output_data_path, exist_ok=True)
    aligned_image.save(f'{output_data_path}/{real_name}.jpeg')
    print("image data have beend aligned.")


def export_updated_pickle(img_name, checkpoints_dir="./results/checkpoints"):
    print("Exporting large updated pickle based off new generator and ffhq.pkl")
    
    with open(f'{checkpoints_dir}/stylegan2_custom_{img_name}.pt', 'rb') as f_new:
        new_G = torch.load(f_new).cuda()
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        d = pickle.load(f)
        old_G = d['G_ema'].cuda() ## tensor
        old_D = d['D'].eval().requires_grad_(False).cpu()

    tmp = {}
    tmp['G'] = old_G.eval().requires_grad_(False).cpu()
    tmp['G_ema'] = new_G.eval().requires_grad_(False).cpu()
    tmp['D'] = old_D
    tmp['training_set_kwargs'] = None
    tmp['augment_pipe'] = None

    with open(f'{checkpoints_dir}/stylegan2_custom_{img_name}.pkl', 'wb') as f:
        pickle.dump(tmp, f)


def pti_run(real_img_path):
    img_name = os.path.basename(real_img_path).split('.')[0]
    pre_process_image(real_img_path)
    print("-------------Run PTI-------------")
    run_PTI(img_processed_path=f"{output_data_path}/{img_name}.jpeg")
    export_updated_pickle(img_name)


if __name__ == "__main__":
    # config setup
    parser = ArgumentParser()
    parser.add_argument('--raw_img_path', type=str)
    parser.add_argument('--max_pti_steps', type=int, default=350)
    config = parser.parse_args()
    real_img_path = config.raw_img_path
    max_pti_steps = config.max_pti_steps
    
    img_name = os.path.basename(real_img_path).split('.')[0]
    pre_process_image(real_img_path)

    run_PTI(img_processed_path=f"{output_data_path}/{img_name}.jpeg", max_pti_steps=max_pti_steps)
    export_updated_pickle(img_name)
