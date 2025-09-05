import replicate
from youzu.image_model import run_youzu
from datetime import datetime

token = "none"
client = replicate.Client(api_token=token)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"interior_{time_str}.png"
input_image = r"C:\Users\USER\Downloads\test17.png"

prompt = "Modern minimalist kitchen interior with clearly separated elements..."
negative_prompt = "blurred appliances, overlapping cabinets..."

run_youzu(
    client, 
    input_image, 
    prompt, 
    filename, 
    negative_prompt=negative_prompt,
    num_inference_steps=30,    # 낮추면 빠르지만 품질 낮음
    guidance_scale=8.0,        # 높이면 프롬프트 준수도 높음  
    prompt_strength=0.7,       # 낮추면 원본 더 보존
    seed=12345                 # 고정하면 재현 가능
)
