import Rhino


import io
import System
from System.Drawing import Imaging, Bitmap
from System.IO import MemoryStream
from System.Drawing import Image as SystemImage
from System import Array, Byte

import io
import requests
from PIL import Image
from io import BytesIO
import os
import json


def image_detection_by_replicate(client, img_file, count=5, confidence_thrshold=0.45):
    queries = [
        # 가구
        "sofa",
        "armchair",
        "dining table",
        "coffee table",
        "side table",
        "bed",
        "bunk bed",
        "desk",
        "office chair",
        "wardrobe",
        "dresser",
        "bookshelf",
        "cabinet",
        "drawer",
        # 주방 가구 및 가전
        "kitchen cabinet",
        "sink",
        "stove",
        "oven",
        "microwave",
        "refrigerator",
        "dishwasher",
        "kitchen island",
        "range hood",
        # 조명
        "ceiling light",
        "pendant light",
        "chandelier",
        "floor lamp",
        "table lamp",
        "wall sconce",
        # 소품 / 장식
        "curtain",
        "rug",
        "mirror",
        "painting",
        "clock",
        "plant",
        "vase",
        "pillow",
        "blanket",
        # 욕실
        "bathtub",
        "shower",
        "toilet",
        "washbasin",
        "mirror cabinet",
        "towel rack",
        # 전자기기
        "television",
        "monitor",
        "speaker",
        "air conditioner",
        "heater",
        "fan",
    ]
    dino_output = client.run(
        "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
        input={
            "image": img_file,
            "query": ",".join(queries),
            "box_threshold": 0.23,
            "text_threshold": 0.2,
            "show_visualisation": False,
        },
    )

    print("dino_output:", dino_output)

    # dino_output이 문자열 리스트라면, 각 요소를 json/dict로 변환
    parsed_output = []
    if isinstance(dino_output, list):
        for obj in dino_output:
            if isinstance(obj, dict):
                parsed_output.append(obj)
            elif isinstance(obj, str):
                try:
                    parsed_output.append(json.loads(obj))
                except Exception as e:
                    print("Failed to parse object:", obj)
    elif isinstance(dino_output, dict):
        parsed_output = [dino_output]
    else:
        print("dino_output is not a list or dict!")

    print("Parsed objects:", parsed_output)

    if not parsed_output:
        print("No objects detected. Check the model output above.")

    # 2. 원본 이미지 로드
    img = Image.open(img_file).convert("RGBA")
    cropped_images = []
    detections = parsed_output[0].get("detections", [])
    detections_filtered = [
        det for det in detections if det.get("confidence") > confidence_thrshold
    ]
    if len(detections_filtered) < 3:
        detections_filtered = sorted(
            detections, key=lambda det: det.get("confidence"), reverse=True
        )[:5]
    # 3. 각 객체별로 crop 후 remove-background 모델에 전달
    for idx, obj in enumerate(detections_filtered):
        box = obj.get("bbox")
        if not box:
            continue
        x1, y1, x2, y2 = map(int, box)
        cropped = img.crop((x1, y1, x2, y2))

        # 임시로 crop 이미지를 메모리에 저장
        buf = BytesIO()
        cropped.save(buf, format="PNG")
        buf.seek(0)

        # Replicate remove-background 모델에 업로드 (바이너리 사용)
        remove_output = client.run(
            "bria/remove-background",
            input={
                "image": buf,
                "content_moderation": False,
                "preserve_partial_alpha": True,
            },
        )

        # 결과 PNG 다운로드 및 저장
        out_url = remove_output  # 항상 문자열 URL로 반환됨
        out_img = requests.get(out_url)
        out_path = f"object_{idx+1}.png"
        cropped_images.append(out_img.content)
        print(f"Saved: {out_path}")
    return cropped_images


def bytes_to_bytesio(data: bytes) -> io.BytesIO:
    """Convert bytes object to BytesIO object."""
    return io.BytesIO(data)


def bitmap_to_bytesio(bmp, format=Imaging.ImageFormat.Png) -> io.BytesIO:
    from PIL import Image

    bmp.Save("tmp.png", format)
    img = Image.open("tmp.png").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    # 파일명 지정(업로더가 MIME 추정하기 쉬움)
    buf.name = "input.jpg"
    return buf


def python_byte_to_Dotnet_bitmap(python_byte):

    # Python bytes → .NET byte[] 변환
    net_bytes = Array[Byte](python_byte)

    # 이제 MemoryStream 생성
    ms = MemoryStream(net_bytes)

    # 이미지 로드
    img = SystemImage.FromStream(ms, True, True)
    bmp = Bitmap(img)  # 독립된 Bitmap 복제

    # 리소스 정리
    img.Dispose()
    ms.Close()

    return bmp
