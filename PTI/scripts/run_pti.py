import os
from configs import global_config
from training.coaches.single_id_coach import SingleIDCoach


def run_PTI(img_processed_path, max_pti_steps):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    coach = SingleIDCoach(img_processed_path, max_pti_steps)
    coach.train()


