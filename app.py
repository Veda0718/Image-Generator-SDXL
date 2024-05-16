import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torch
from typing import Tuple

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + negative

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt, negative_prompt, width, height, guidance_scale, style_name=None):
    seed = random.randint(0,4294967295)

    generator = torch.Generator().manual_seed(seed)

    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    image = [pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale,
        width = width,
        height = height,
        generator = generator
    ).images[0] for _ in range(4)]

    return image

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
    "A serious capybara at work, wearing a suit",
    'A Squirtle fine dining with a view to the London Eye',
    'a graffiti of a robot serving meals to people',
    'a beautiful cabin in Attersee, Austria, 3d animation style',
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 1000px;
    padding-top: 20px;
    text-align: center;
}
.header {
            margin: 10px auto 10px auto;
            text-align: center;
            max-width: 600px;
        }
#example-container {
    max-width: 1000px;
    margin: 0 auto;
}
.footer {
            margin: 25px auto 45px auto;
            text-align: center;
            max-width: 600px;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
        }
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
      gr.HTML(
            """
                <div class="header">
                  <h1>Welcome to Metamorph: Your Creative Gateway</h1>
                    <h4>
                    Transform your words into stunning visuals with our advanced AI-powered Text-to-Image generator
                    </h4>
                </div>
           """)
      gr.Markdown(f"""
        Currently running on {power_device}.
        """)
    with gr.Row(elem_id="col-container"):
      # Left column
      with gr.Column(scale=1,elem_id="left-container"):
        with gr.Row():
          prompt = gr.Text(
              label="Prompt",
              show_label=False,
              max_lines=1,
              placeholder="Enter your prompt",
              container=False,
          )
          run_button = gr.Button("Generate", scale=0)

        with gr.Accordion("Advanced Settings", open=True):

          negative_prompt = gr.Textbox(
              label="Negative prompt",
              show_label=False,
              max_lines=1,
              placeholder="Enter a negative prompt",
              elem_id="negative-prompt-text-input",
          )

          style_selection = gr.Radio(
              show_label=True,
              container=True,
              interactive=True,
              choices=STYLE_NAMES,
              value=DEFAULT_STYLE_NAME,
              label="Image Style",
          )

          with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )

            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )

          with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.0,
                maximum=50.0,
                step=0.1,
                value=10,
            )

      # Right column
      with gr.Column(scale=1, elem_id="right-container"):
        result = gr.Gallery(label="Results", show_label=False, format="png", show_share_button=False, height=475)

    gr.Examples(
            elem_id="example-container",
            examples = examples,
            inputs = [prompt]
        )

    gr.HTML(
            """
                <div class="footer">
                    <p>
This application harnesses the cutting-edge Stable Diffusion XL (SDXL) model by <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">StabilityAI</a>, offering unparalleled text-to-image generation, while acknowledging potential biases and content considerations outlined in the model card.</p>
                    </p>
                </div>
           """
    )

    run_button.click(
        fn = infer,
        inputs = [prompt, negative_prompt, width, height, guidance_scale, style_selection],
        outputs = [result]
    )

demo.queue().launch()