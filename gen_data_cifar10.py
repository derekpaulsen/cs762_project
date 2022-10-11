import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import time
from pathlib import Path
from tempfile import mkstemp
from tqdm import tqdm
from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument('--out_dir', required=True, type=str)
argp.add_argument('--batch_size', default=2, type=int)
argp.add_argument('--images_per_class', required=True, type=int)
argp.add_argument('--reps', default=1, type=int)
argp.add_argument('--image_size', default=512, type=int)
argp.add_argument('--guidance_scale', default=7.5, type=float)
argp.add_argument('--num_inference_steps', default=50, type=int)


AUTH_TOKEN=open('./auth_token').read().strip()
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"


CIFAR_10_LABELS = [
        'airplane',
        'automobile', 
        'bird', 
        'cat', 
        'deer', 
        'dog', 
        'frog', 
        'horse', 
        'ship', 
        'truck'
]



def filter_nsfw(out):
    valid_images = [i for i, nsfw in zip(out['images'], out['nsfw_content_detected']) if not nsfw]
    return valid_images

def save_images(out_dir, prompt, images):
    out_dir = out_dir / prompt
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'saving images to {out_dir}')
    for i in images:
        out_path = mkstemp(prefix=prompt + '_', suffix='.png', dir=out_dir)[1]
        i.save(out_path)

def generate_images(pipe, prompt, count, batch_size, kwargs):
    images = []
    prompt = [prompt] * batch_size
    with tqdm(total=count) as pbar:
        while len(images) < count:
            out = pipe(prompt, **kwargs)
            safe = filter_nsfw(out)
            images += safe
            pbar.update(len(safe))

    return images


def generate_and_save_images(pipe, out_dir, prompt, count, batch_size, kwargs):
    print(f'generating images for prompt "{prompt}"')
    images = generate_images(pipe, prompt, count, batch_size, kwargs)
    save_images(out_dir, prompt, images)




def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, revision="fp16", use_auth_token=AUTH_TOKEN)
    pipe = pipe.to(DEVICE)

    kwargs = {
            'height' : args.image_size,
            'width' : args.image_size,
            'guidance_scale' : args.guidance_scale,
            'num_inference_steps' : args.num_inference_steps
    }

    start_t = time.time()
    with autocast("cuda"):
        for i in range(args.reps):
            for prompt in CIFAR_10_LABELS:
                generate_and_save_images(pipe, out_dir, prompt, args.images_per_class, args.batch_size, kwargs)

    elapsed = time.time() - start_t
    print(f'done : {elapsed}')


if __name__ == '__main__':
    main(argp.parse_args())
