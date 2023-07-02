import gradio as gr
import random
import time
import tqdm
import shutil
from uuid import uuid4

with gr.Blocks() as demo:
    with gr.Row():
        text = gr.Textbox()
        textb = gr.Textbox()
    with gr.Row():
        load_set_btn = gr.Button("Load Set")
        load_nested_set_btn = gr.Button("Load Nested Set")
        load_random_btn = gr.Button("Load Random")
        clean_imgs_btn = gr.Button("Clean Images")
        wait_btn = gr.Button("Wait")
        do_all_btn = gr.Button("Do All")
        track_tqdm_btn = gr.Button("Bind TQDM")
        bind_internal_tqdm_btn = gr.Button("Bind Internal TQDM")

    text2 = gr.Textbox()

    # track list
    def load_set(text, text2, progress=gr.Progress()):
        imgs = [None] * 24
        for img in progress.tqdm(imgs, desc="Loading from list"):
            time.sleep(0.1)
        return "done"
    load_set_btn.click(load_set, [text, textb], text2)


if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch()
