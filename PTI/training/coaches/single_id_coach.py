import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from torchvision.transforms import transforms
from PIL import Image


class SingleIDCoach(BaseCoach):

    def __init__(self, img_processed_path, max_pti_steps, use_wandb=False):
        super().__init__(use_wandb)
        self.img_processed_path = img_processed_path
        self.max_pti_steps = max_pti_steps
        
    def preprocess_img(self):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        img = Image.open(self.img_processed_path).convert('RGB')
        img = trans(img)
        return img

    def train(self):
        self.img_name = os.path.basename(self.img_processed_path).split('.')[0]
        img = self.preprocess_img()

        w_path_dir = f'{paths_config.embedding_base_dir}'
        os.makedirs(w_path_dir, exist_ok=True)

        use_ball_holder = True

        self.restart_training()

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, self.img_name)

        elif not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(img, self.img_name)

        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        w_pivot = w_pivot.to(global_config.device)

        torch.save(w_pivot, f'{w_path_dir}/embedding.pt')
        log_images_counter = 0
        real_images_batch = img.to(global_config.device).unsqueeze(dim=0)
        # print(real_images_batch.shape)

        for i in tqdm(range(self.max_pti_steps)):

            generated_images = self.forward(w_pivot)
            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, self.img_name,
                                                            self.G, use_ball_holder, w_pivot)

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

            if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                log_images_from_w([w_pivot], self.G, [self.img_name])

            global_config.training_step += 1
            log_images_counter += 1


        os.makedirs(paths_config.checkpoints_dir, exist_ok=True)
        torch.save(self.G, f'{paths_config.checkpoints_dir}/stylegan2_custom_{self.img_name}.pt')
