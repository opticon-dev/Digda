import requests

import os
from PIL import Image
import io


def upscale(client, image: io.BytesIO):

    # Step2 폴더의 모든 이미지 파일 처리
    # 이미지 업스케일링
    output = client.run(
        "bria/increase-resolution",
        input={
            "sync": True,
            "image": image,
            "preserve_alpha": True,
            "desired_increase": 4,
            "content_moderation": False,
        },
    )

    return output.read()


def run_flux_nano_banana(client, image, prompt):
    output = client.run(
        "google/nano-banana",
        input={"prompt": prompt, "image_input": [image], "output_format": "jpg"},
    )

    # To access the file URL:
    # => "http://example.com"

    # To write the file to disk:

    return output.read()


def run_flux_kontext_dev(client, image: io.BytesIO, prompt):
    output = client.run(
        "black-forest-labs/flux-kontext-dev",
        input={
            "prompt": prompt,
            "go_fast": True,
            "guidance": 2.5,
            "input_image": image,
            "aspect_ratio": "match_input_image",
            "output_format": "jpg",
            "output_quality": 80,
            "num_inference_steps": 30,
        },
    )

    return output.read()


def run_flux_dev(
    client,
    image: io.BytesIO,
    prompt,
    filename,
    num_inference_steps=50,
    guidance_scale=7.5,
    prompt_strength=0.8,
    seed=None,
):
    """
    Flux-dev 모델을 실행하는 함수
    """
    input = {
        "image": image,
        "prompt": prompt,
        "output_format": "png",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "prompt_strength": prompt_strength,
    }

    if seed is not None:
        input["seed"] = seed

    output = client.run("black-forest-labs/flux-dev", input=input)

    for index, item in enumerate(output):
        if index == 0:
            with open(filename, "wb") as file:
                file.write(item.read())
            return item.read()


def run_youzu(
    client,
    image,
    prompt,
    filename,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    prompt_strength=0.8,
    seed=None,
):

    input = {
        "image": image,
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "prompt_strength": prompt_strength,
    }

    if negative_prompt:
        input["negative_prompt"] = negative_prompt

    if seed is not None:
        input["seed"] = seed

    output = client.run(
        "youzu/stable-interiors-v2:4836eb257a4fb8b87bac9eacbef9292ee8e1a497398ab96207067403a4be2daf",
        input=input,
    )

    response = requests.get(output)
    with open(filename, "wb") as file:
        file.write(response.content)
    return response.content
