import replicate
import requests
import os

token = None
client = replicate.Client(api_token=token)

input = {
    "image": "https://replicate.delivery/pbxt/MyNhSZa0UVCZCAXKswYb6PSzwQLAT7ou2KyzwsBHRMydQm3s/bedroom_2.jpg",
    "prompt": "a fully furnished Scandinavian living room with one light wood coffee table, a neutral-toned fabric sofa on the left side of the wall, soft wool throws and cushions, white walls, minimalist shelving, a pendant lamp overhead, and minimal decor, with natural light streaming through sheer linen curtains",
}

output = client.run(
    "youzu/stable-interiors-v2:4836eb257a4fb8b87bac9eacbef9292ee8e1a497398ab96207067403a4be2daf",
    input=input,
)

existing_files = [
    f for f in os.listdir(".") if f.startswith("interior_") and f.endswith(".png")
]
next_number = len(existing_files) + 1
filename = f"interior_{next_number}.png"

response = requests.get(output)
with open(filename, "wb") as file:
    file.write(response.content)
