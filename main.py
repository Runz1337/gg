import os
os.system("pip install -r requirements.txt")

import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("darkstorm2150/Protogen_v5.8_Official_Release", torch_dtype=torch.float16)  
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
from flask import Flask, send_file
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)

from flask import request
generator = torch.Generator("cuda").manual_seed(1024)

@app.route('/')
def hello_world():
    prompt=request.args.get('dream')
    promt=prompt.replace("+"," ")
    neg=neg.replace("+"," ")
    neg=request.args.get('neg')

    image = pipe(prompt,negative_prompt=neg,num_inference_steps=130, guidance_scale = 10,generator=generator).images[0]
    image.save("sd_image.png")

    return send_file("sd_image.png", mimetype='image/gif')   #return res





app.run()
