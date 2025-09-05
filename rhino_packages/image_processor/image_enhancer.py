import requests


def run_youzu(
    client,
    image_path,
    prompt,
    filename,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    prompt_strength=0.8,
    seed=None,
):
    if isinstance(image_path, str):
        image = open(image_path, "rb")
    else:
        image = image_path
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
