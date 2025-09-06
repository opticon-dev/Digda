import openai
import base64
import json
from PIL import Image
import os
import replicate
import requests
import time
import shutil
from datetime import datetime
import io
from typing import Union, Tuple

# =============================================================================
# 기본 유틸리티 함수들
# =============================================================================


def encode_image_to_base64(image_path):
    """이미지 파일을 Base64 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_buffer_to_base64(image_buffer):
    return base64.b64encode(image_buffer).decode("utf-8")


def pil_to_filelike(pil_img, format="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf


def open_image_by_type(image_path: Union[str, bytes, io.BytesIO]):
    if isinstance(image_path, str):
        # 파일 경로
        _img = None
        with Image.open(image_path) as img:
            _img = img
        return _img

    elif isinstance(image_path, bytes):
        # raw bytes
        img = Image.open(io.BytesIO(image_path))
        return img
    elif isinstance(image_path, io.BytesIO):
        # 이미 BytesIO 객체
        img = Image.open(image_path)
        return img
    else:
        raise NotImplementedError


class ImageProcessor:
    def __init__(self, OPENAI_API_KEY, REPLICATE_API_TOKEN):
        self.open_ai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    def process_1(self, image):
        # 단계 1: 가구 인식 및 크롭
        print("\n🔥 [단계 1] 가구 인식 및 크롭 시작...")
        step1_result = FurnitureCropper(self.open_ai_client).process(image)

        if not step1_result:
            print("❌ 단계 1 실패: 가구 인식에 실패했습니다.")
            return None

        print(
            f"✅ 단계 1 완료: {len(step1_result['detected_furniture'])}개 가구 처리됨"
        )
        # 단계 2: 배경 제거
        print("\n🔥 [단계 2] Bria 배경 제거 시작...")
        step2_result = BackgroundRemover(self.replicate_client).process(
            step1_result.get("cropped_images")
        )
        return step2_result

    def process_2(self, selected_images):
        print("\n🔥 [단계 3] HunYuan3D 3D 변환 시작...")
        processed_files = ImgToModeling(self.replicate_client).process(selected_images)

        if not processed_files:
            print("❌ 단계 3 실패: 3D 변환에 실패했습니다.")
            return None

        return processed_files


# =============================================================================
# 단계 1: GPT API를 사용한 가구 인식 및 중심 맞춤 크롭 (중복 제거 버전)
# =============================================================================


def calculate_box_area(box):
    """박스의 면적 계산"""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def calculate_overlap_ratio(box1, box2):
    """두 박스의 겹치는 비율 계산 (작은 박스 기준)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 겹치는 영역 계산
    overlap_x1 = max(x1_1, x1_2)
    overlap_y1 = max(y1_1, y1_2)
    overlap_x2 = min(x2_1, x2_2)
    overlap_y2 = min(y2_1, y2_2)

    # 겹치는 영역이 있는지 확인
    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
        return 0.0

    # 겹치는 면적
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)

    # 더 작은 박스의 면적
    area1 = calculate_box_area(box1)
    area2 = calculate_box_area(box2)
    smaller_area = min(area1, area2)

    # 작은 박스 기준 겹치는 비율
    return overlap_area / smaller_area if smaller_area > 0 else 0.0


class FurnitureRebuilder:
    GPT_MODEL = "gpt-4o"

    def __init__(self, open_ai_client):
        self.open_ai_client = open_ai_client

    def write_prompt(self, img_width, img_height):
        # GPT 프롬프트 (주요 가구 중심)
        return f"""
당신은 인테리어 전문가입니다. 이 거실 사진에서 주요 가구들을 정확히 찾고 그 가구가 일부만 보이더라도 나머지 부분을 상상해서 완성하세요.

이미지 크기: {img_width} x {img_height} 픽셀

우선순위별 가구 목록:

🏠 대형 가구 (최우선):
- 소파, 쇼파 (sofa, couch, sectional)
- 큰 테이블 (dining table, large coffee table)
- 큰 선반/책장 (bookshelf, large cabinet)
- 침대 (bed, mattress)

🪑 중형 가구:
- 의자 (chair, armchair, recliner)
- 작은 테이블 (side table, coffee table)
- TV/모니터 (television, monitor)
- 사다리 (ladder, step)

🧸 소형 가구/소품:
- 조명 (lamp, floor lamp)
- 식물/화분 (plant, pot)
- 장식품 (decoration, vase)
- 쿠션 (cushion, pillow)

중요 지침:
1. 큰 가구를 우선적으로 인식
2. 각 가구는 충분히 큰 크기여야 함 (최소 50x50 픽셀)
3. 명확하게 구분되는 개별 가구만 포함
4. 애매하거나 일부만 보이는 것은 제외

이미지 형식으로만 응답:
설명 없이 반환하세요.
        """

    def process(self, image_path):
        """단계 1: 중복 제거된 가구 인식 및 중심 맞춤 크롭"""
        print("=" * 70)
        print("🚀 단계 1: GPT 가구 인식 + 중복 제거 + 중심 맞춤 크롭")
        print("=" * 70)

        # 1. 가구 인식 (중복 제거 포함)
        print("\n📍 1단계: 가구 인식 및 중복 제거")
        furniture_list = self._detect_furniture_with_gpt_filtered(image_path)

    def _detect_furniture_with_gpt_filtered(self, image_path):
        """GPT API를 사용하여 가구를 인식하고 중복 제거"""
        print(f"🔍 이미지 분석 시작: {image_path}")

        if isinstance(image_path, str):
            # 파일 경로
            with Image.open(image_path) as img:
                img_width, img_height = img.width, img.height
            base64_image = encode_image_to_base64(image_path)

        elif isinstance(image_path, bytes):
            # raw bytes
            img = Image.open(io.BytesIO(image_path))
            img_width, img_height = img.width, img.height
            base64_image = image_buffer_to_base64(image_path)

        elif isinstance(image_path, io.BytesIO):
            # 이미 BytesIO 객체
            img = Image.open(image_path)
            img_width, img_height = img.width, img.height
            # BytesIO → bytes 변환해서 base64 인코딩
            base64_image = image_buffer_to_base64(image_path.getvalue())

        else:
            raise TypeError(f"지원하지 않는 타입: {type(image_path)}")

        # GPT API 호출
        response = self.open_ai_client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.write_prompt(img_width, img_height),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=3000,
            temperature=0.1,
        )

        # 응답 처리
        response_text = response.choices[0].message.content.strip()
        print(f"📝 GPT 응답 받음 (길이: {len(response_text)}자)")

        # JSON 추출
        json_str = response_text.replace("```json", "").replace("```", "").strip()
        print(json_str)

    def _filter_overlapping_furniture(self, furniture_list, overlap_threshold=0.7):
        """중복되는 가구 제거 (큰 가구 우선, 겹치는 비율이 높은 작은 가구 제거)"""
        print(f"\n🔍 중복 가구 필터링 시작 (임계값: {overlap_threshold*100}%)")

        # 면적 기준으로 내림차순 정렬 (큰 가구부터)
        sorted_furniture = sorted(
            furniture_list, key=lambda x: calculate_box_area(x["box"]), reverse=True
        )

        filtered_furniture = []
        removed_count = 0

        for current_furniture in sorted_furniture:
            current_box = current_furniture["box"]
            current_area = calculate_box_area(current_box)
            current_name = current_furniture["name"]

            is_overlapping = False

            # 이미 선택된 가구들과 비교
            for selected_furniture in filtered_furniture:
                selected_box = selected_furniture["box"]
                selected_name = selected_furniture["name"]

                overlap_ratio = calculate_overlap_ratio(current_box, selected_box)

                if overlap_ratio > overlap_threshold:
                    print(
                        f"❌ {current_name} (면적: {current_area}) - {selected_name}와 {overlap_ratio:.1%} 겹침"
                    )
                    is_overlapping = True
                    removed_count += 1
                    break

            if not is_overlapping:
                filtered_furniture.append(current_furniture)
                print(f"✅ {current_name} (면적: {current_area}) - 유지")

        print(
            f"📊 필터링 결과: {len(furniture_list)}개 → {len(filtered_furniture)}개 (제거: {removed_count}개)"
        )
        return filtered_furniture

    def _categorize_furniture_by_size(self, furniture_list, img_width, img_height):
        """가구를 크기별로 분류"""
        total_image_area = img_width * img_height

        large_furniture = []  # 큰 가구 (이미지의 3% 이상)
        medium_furniture = []  # 중간 가구 (이미지의 1-3%)
        small_furniture = []  # 작은 가구 (이미지의 1% 미만)

        for furniture in furniture_list:
            area = calculate_box_area(furniture["box"])
            area_ratio = area / total_image_area

            furniture["area"] = area
            furniture["area_ratio"] = area_ratio

            if area_ratio >= 0.03:  # 3% 이상
                large_furniture.append(furniture)
            elif area_ratio >= 0.01:  # 1-3%
                medium_furniture.append(furniture)
            else:  # 1% 미만
                small_furniture.append(furniture)

        print(f"📏 크기별 분류:")
        print(f"   🏠 큰 가구 ({len(large_furniture)}개): 이미지의 3% 이상")
        print(f"   🪑 중간 가구 ({len(medium_furniture)}개): 이미지의 1-3%")
        print(f"   🧸 작은 가구 ({len(small_furniture)}개): 이미지의 1% 미만")

        return large_furniture, medium_furniture, small_furniture

    def calculate_size_analysis(self, furniture_list, img_width, img_height):
        """상대적 크기 분석 계산"""
        if not furniture_list:
            return {}

        print("\n📊 상대적 크기 분석 중...")

        # 기본 정보
        total_image_area = img_width * img_height
        largest_furniture_area = max(
            furniture.get("area", 0) for furniture in furniture_list
        )

        # 각 가구의 상대적 크기 정보 계산
        furniture_size_comparison = []

        for furniture in furniture_list:
            name = furniture["name"]
            area = furniture.get("area", 0)

            # 비율 계산
            area_ratio_to_image = (area / total_image_area) * 100
            area_ratio_to_largest = (
                (area / largest_furniture_area) * 100
                if largest_furniture_area > 0
                else 0
            )

            # 크기 등급 결정
            if area_ratio_to_image >= 3.0:
                size_rank = "대형"
            elif area_ratio_to_image >= 1.0:
                size_rank = "중형"
            else:
                size_rank = "소형"

            # 상대 점수 (가장 큰 가구 = 100점)
            relative_size_score = round(area_ratio_to_largest, 1)

            size_info = {
                "name": name,
                "absolute_area": area,
                "area_ratio_to_image": f"{area_ratio_to_image:.2f}%",
                "area_ratio_to_largest": f"{area_ratio_to_largest:.1f}%",
                "size_rank": size_rank,
                "relative_size_score": relative_size_score,
            }

            furniture_size_comparison.append(size_info)
            print(
                f"  📏 {name:15s}: {area:,}px² ({area_ratio_to_image:.2f}%, {size_rank}, 점수: {relative_size_score})"
            )

        # 크기순으로 정렬
        furniture_size_comparison.sort(key=lambda x: x["absolute_area"], reverse=True)

        size_analysis = {
            "image_dimensions": {"width": img_width, "height": img_height},
            "total_image_area": total_image_area,
            "largest_furniture_area": largest_furniture_area,
            "furniture_count": len(furniture_list),
            "size_distribution": {
                "large": len(
                    [
                        f
                        for f in furniture_list
                        if (f.get("area", 0) / total_image_area) >= 0.03
                    ]
                ),
                "medium": len(
                    [
                        f
                        for f in furniture_list
                        if 0.01 <= (f.get("area", 0) / total_image_area) < 0.03
                    ]
                ),
                "small": len(
                    [
                        f
                        for f in furniture_list
                        if (f.get("area", 0) / total_image_area) < 0.01
                    ]
                ),
            },
            "furniture_size_comparison": furniture_size_comparison,
        }

        print(
            f"✅ 크기 분석 완료: 대형 {size_analysis['size_distribution']['large']}개, 중형 {size_analysis['size_distribution']['medium']}개, 소형 {size_analysis['size_distribution']['small']}개"
        )

        return size_analysis

    def crop_furniture_centered_filtered(
        self, image_path, furniture_list, output_dir="furniture_crops_filtered"
    ):
        """각 가구를 중심에 맞춰서 크롭 (필터링된 버전)"""
        # 출력 디렉토리 생성

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 이미지 열기
        img = open_image_by_type(image_path)
        img_width, img_height = img.size
        if isinstance(img, str):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = "IMG_FROM_RHINO"
        print(f"🎯 {len(furniture_list)}개 가구를 중심 맞춤 크롭합니다...")

        # 우선순위별로 정렬 (면적 기준 내림차순)
        sorted_furniture = sorted(
            furniture_list, key=lambda x: x.get("area", 0), reverse=True
        )

        cropped_images = []

        for i, furniture in enumerate(sorted_furniture):
            name = furniture["name"]
            category = furniture.get("category", "medium")
            priority = furniture.get("priority", "medium")
            x1, y1, x2, y2 = furniture["box"]
            area = furniture.get("area", 0)

            try:
                # 원본 가구 크기
                furniture_width = x2 - x1
                furniture_height = y2 - y1
                furniture_center_x = (x1 + x2) // 2
                furniture_center_y = (y1 + y2) // 2

                # 크롭 크기 결정 (카테고리별 여백 조정)
                if category == "large":
                    margin_ratio = 0.05  # 큰 가구는 5% 여백
                elif category == "medium":
                    margin_ratio = 0.1  # 중간 가구는 10% 여백
                else:
                    margin_ratio = 0.15  # 작은 가구는 15% 여백

                margin_x = int(furniture_width * margin_ratio)
                margin_y = int(furniture_height * margin_ratio)
                crop_width = furniture_width + 2 * margin_x
                crop_height = furniture_height + 2 * margin_y

                # 중심 맞춤 크롭 좌표 계산
                crop_x1 = max(0, furniture_center_x - crop_width // 2)
                crop_y1 = max(0, furniture_center_y - crop_height // 2)
                crop_x2 = min(img_width, crop_x1 + crop_width)
                crop_y2 = min(img_height, crop_y1 + crop_height)

                # 경계 조정
                if crop_x2 - crop_x1 < crop_width:
                    crop_x1 = max(0, crop_x2 - crop_width)
                if crop_y2 - crop_y1 < crop_height:
                    crop_y1 = max(0, crop_y2 - crop_height)

                # 이미지 크롭
                cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # 파일명 정리 (우선순위 포함)
                safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                filename = f"{base_name}_{i+1:02d}_{priority}_{safe_name}.png"
                filepath = os.path.join(output_dir, filename)

                # 저장
                cropped_img.save(filepath)

                # 결과 정보 저장
                crop_info = {
                    "original_furniture": furniture,
                    "crop_coordinates": [crop_x1, crop_y1, crop_x2, crop_y2],
                    "crop_size": f"{crop_x2-crop_x1}x{crop_y2-crop_y1}",
                    "filename": filename,
                    "filepath": filepath,
                }
                cropped_images.append(crop_info)

                area_ratio = area / (img_width * img_height) * 100
                print(
                    f"✅ {i+1:2d}. {name:15s} ({priority:6s}) → {filename} ({crop_x2-crop_x1}x{crop_y2-crop_y1}, {area_ratio:.1f}%)"
                )

            except Exception as e:
                print(f"❌ {name} 크롭 실패: {e}")

        print(f"🎉 총 {len(cropped_images)}개 가구 크롭 완료!")
        print(f"📁 저장 위치: {output_dir}/")

        return cropped_images


class FurnitureCropper:
    GPT_MODEL = "gpt-4o"

    def __init__(self, open_ai_client):
        self.open_ai_client = open_ai_client

    def write_prompt(self, img_width, img_height):
        # GPT 프롬프트 (주요 가구 중심)
        return f"""
당신은 인테리어 전문가입니다. 이 거실 사진에서 주요 가구들을 정확히 찾아주세요.

이미지 크기: {img_width} x {img_height} 픽셀

우선순위별 가구 목록:

🏠 대형 가구 (최우선):
- 소파, 쇼파 (sofa, couch, sectional)
- 큰 테이블 (dining table, large coffee table)
- 큰 선반/책장 (bookshelf, large cabinet)
- 침대 (bed, mattress)

🪑 중형 가구:
- 의자 (chair, armchair, recliner)
- 작은 테이블 (side table, coffee table)
- TV/모니터 (television, monitor)
- 사다리 (ladder, step)

🧸 소형 가구/소품:
- 조명 (lamp, floor lamp)
- 식물/화분 (plant, pot)
- 장식품 (decoration, vase)
- 쿠션 (cushion, pillow)

중요 지침:
1. 큰 가구를 우선적으로 인식
2. 각 가구는 충분히 큰 크기여야 함 (최소 50x50 픽셀)
3. 명확하게 구분되는 개별 가구만 포함
4. 애매하거나 일부만 보이는 것은 제외

JSON 형식으로만 응답:
{{
"furniture_list": [
    {{
    "name": "구체적인_가구_이름",
    "category": "large/medium/small", 
    "priority": "high/medium/low",
    "box": [x1, y1, x2, y2],
    "confidence": "high/medium/low"
    }}
]
}}

좌표 규칙:
- [x1, y1] = 왼쪽 아래 모서리
- [x2, y2] = 오른쪽 위 모서리
- 0 ≤ x1 < x2 ≤ {img_width}
- 0 ≤ y1 < y2 ≤ {img_height}

설명 없이 JSON만 반환하세요.
        """

    def process(self, image_path):
        """단계 1: 중복 제거된 가구 인식 및 중심 맞춤 크롭"""
        print("=" * 70)
        print("🚀 단계 1: GPT 가구 인식 + 중복 제거 + 중심 맞춤 크롭")
        print("=" * 70)

        # 1. 가구 인식 (중복 제거 포함)
        print("\n📍 1단계: 가구 인식 및 중복 제거")
        furniture_list = self._detect_furniture_with_gpt_filtered(image_path)

        if not furniture_list:
            print("❌ 가구를 찾지 못했습니다.")
            return None

        # 2. 중심 맞춤 크롭 및 이미지 저장
        print("\n📍 2단계: 중심 맞춤 크롭")
        cropped_images = self.crop_furniture_centered_filtered(
            image_path, furniture_list
        )

        # 3. 상대적 크기 분석 추가
        print("\n📍 3단계: 상대적 크기 분석")
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        size_analysis = self.calculate_size_analysis(
            furniture_list, img_width, img_height
        )

        # 4. 결과 저장 (크기 분석 포함)
        result_data = {
            "detected_furniture": furniture_list,
            "cropped_images": cropped_images,
            "size_analysis": size_analysis,  # 새로 추가된 크기 분석 정보
            "summary": {
                "total_detected": len(furniture_list),
                "successfully_cropped": len(cropped_images),
                "large_furniture": len(
                    [f for f in furniture_list if f.get("category") == "large"]
                ),
                "medium_furniture": len(
                    [f for f in furniture_list if f.get("category") == "medium"]
                ),
                "small_furniture": len(
                    [f for f in furniture_list if f.get("category") == "small"]
                ),
            },
        }

        print("\n" + "=" * 70)
        print(f"✅ 단계 1 완료!")
        print(f"📊 최종 선택 가구: {len(furniture_list)}개")
        print(f"🖼️  크롭된 이미지: {len(cropped_images)}개")
        print(f"📏 상대적 크기 분석: 포함됨")
        print(f"💾 결과 저장: step1_furniture_detection_filtered.json")
        print("=" * 70)

        return result_data

    def _detect_furniture_with_gpt_filtered(self, image_path):
        """GPT API를 사용하여 가구를 인식하고 중복 제거"""
        print(f"🔍 이미지 분석 시작: {image_path}")

        if isinstance(image_path, str):
            # 파일 경로
            with Image.open(image_path) as img:
                img_width, img_height = img.width, img.height
            base64_image = encode_image_to_base64(image_path)

        elif isinstance(image_path, bytes):
            # raw bytes
            img = Image.open(io.BytesIO(image_path))
            img_width, img_height = img.width, img.height
            base64_image = image_buffer_to_base64(image_path)

        elif isinstance(image_path, io.BytesIO):
            # 이미 BytesIO 객체
            img = Image.open(image_path)
            img_width, img_height = img.width, img.height
            # BytesIO → bytes 변환해서 base64 인코딩
            base64_image = image_buffer_to_base64(image_path.getvalue())

        else:
            raise TypeError(f"지원하지 않는 타입: {type(image_path)}")

        # GPT API 호출
        response = self.open_ai_client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.write_prompt(img_width, img_height),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=3000,
            temperature=0.1,
        )

        # 응답 처리
        response_text = response.choices[0].message.content.strip()
        print(f"📝 GPT 응답 받음 (길이: {len(response_text)}자)")

        # JSON 추출
        json_str = response_text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(json_str)
            furniture_list = data.get("furniture_list", [])

            print(f"✅ 총 {len(furniture_list)}개 가구 발견")

            # 좌표 검증 및 정리
            valid_furniture = []
            for item in furniture_list:
                name = item.get("name", "unknown")
                category = item.get("category", "medium")
                priority = item.get("priority", "medium")
                box = item.get("box", [])
                confidence = item.get("confidence", "medium")

                if len(box) == 4:
                    x1, y1, x2, y2 = box

                    # 좌표 범위 확인 및 수정
                    x1 = max(0, min(x1, img_width - 1))
                    x2 = max(x1 + 10, min(x2, img_width))
                    y1 = max(0, min(y1, img_height - 1))
                    y2 = max(y1 + 10, min(y2, img_height))

                    # 최소 크기 확인 (50x50 픽셀 이상)
                    if (x2 - x1) >= 50 and (y2 - y1) >= 50:
                        area = (x2 - x1) * (y2 - y1)
                        valid_furniture.append(
                            {
                                "name": name,
                                "category": category,
                                "priority": priority,
                                "box": [x1, y1, x2, y2],
                                "confidence": confidence,
                                "area": area,
                                "size": f"{x2-x1}x{y2-y1}",
                            }
                        )
                        print(
                            f"✅ {name} ({category}/{priority}): [{x1},{y1},{x2},{y2}] - {x2-x1}x{y2-y1}"
                        )
                    else:
                        print(f"⚠️  {name}: 너무 작음 ({x2-x1}x{y2-y1})")
                else:
                    print(f"❌ {name}: 좌표 형식 오류")

            # 크기별 분류
            large_furniture, medium_furniture, small_furniture = (
                self._categorize_furniture_by_size(
                    valid_furniture, img_width, img_height
                )
            )

            # 중복 제거 (큰 가구부터 우선)
            all_furniture = large_furniture + medium_furniture + small_furniture
            filtered_furniture = self._filter_overlapping_furniture(
                all_furniture, overlap_threshold=0.6
            )

            print(f"📋 최종 선택된 가구: {len(filtered_furniture)}개")
            return filtered_furniture

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"응답 내용: {response_text[:500]}...")
            return []

    def _filter_overlapping_furniture(self, furniture_list, overlap_threshold=0.7):
        """중복되는 가구 제거 (큰 가구 우선, 겹치는 비율이 높은 작은 가구 제거)"""
        print(f"\n🔍 중복 가구 필터링 시작 (임계값: {overlap_threshold*100}%)")

        # 면적 기준으로 내림차순 정렬 (큰 가구부터)
        sorted_furniture = sorted(
            furniture_list, key=lambda x: calculate_box_area(x["box"]), reverse=True
        )

        filtered_furniture = []
        removed_count = 0

        for current_furniture in sorted_furniture:
            current_box = current_furniture["box"]
            current_area = calculate_box_area(current_box)
            current_name = current_furniture["name"]

            is_overlapping = False

            # 이미 선택된 가구들과 비교
            for selected_furniture in filtered_furniture:
                selected_box = selected_furniture["box"]
                selected_name = selected_furniture["name"]

                overlap_ratio = calculate_overlap_ratio(current_box, selected_box)

                if overlap_ratio > overlap_threshold:
                    print(
                        f"❌ {current_name} (면적: {current_area}) - {selected_name}와 {overlap_ratio:.1%} 겹침"
                    )
                    is_overlapping = True
                    removed_count += 1
                    break

            if not is_overlapping:
                filtered_furniture.append(current_furniture)
                print(f"✅ {current_name} (면적: {current_area}) - 유지")

        print(
            f"📊 필터링 결과: {len(furniture_list)}개 → {len(filtered_furniture)}개 (제거: {removed_count}개)"
        )
        return filtered_furniture

    def _categorize_furniture_by_size(self, furniture_list, img_width, img_height):
        """가구를 크기별로 분류"""
        total_image_area = img_width * img_height

        large_furniture = []  # 큰 가구 (이미지의 3% 이상)
        medium_furniture = []  # 중간 가구 (이미지의 1-3%)
        small_furniture = []  # 작은 가구 (이미지의 1% 미만)

        for furniture in furniture_list:
            area = calculate_box_area(furniture["box"])
            area_ratio = area / total_image_area

            furniture["area"] = area
            furniture["area_ratio"] = area_ratio

            if area_ratio >= 0.03:  # 3% 이상
                large_furniture.append(furniture)
            elif area_ratio >= 0.01:  # 1-3%
                medium_furniture.append(furniture)
            else:  # 1% 미만
                small_furniture.append(furniture)

        print(f"📏 크기별 분류:")
        print(f"   🏠 큰 가구 ({len(large_furniture)}개): 이미지의 3% 이상")
        print(f"   🪑 중간 가구 ({len(medium_furniture)}개): 이미지의 1-3%")
        print(f"   🧸 작은 가구 ({len(small_furniture)}개): 이미지의 1% 미만")

        return large_furniture, medium_furniture, small_furniture

    def calculate_size_analysis(self, furniture_list, img_width, img_height):
        """상대적 크기 분석 계산"""
        if not furniture_list:
            return {}

        print("\n📊 상대적 크기 분석 중...")

        # 기본 정보
        total_image_area = img_width * img_height
        largest_furniture_area = max(
            furniture.get("area", 0) for furniture in furniture_list
        )

        # 각 가구의 상대적 크기 정보 계산
        furniture_size_comparison = []

        for furniture in furniture_list:
            name = furniture["name"]
            area = furniture.get("area", 0)

            # 비율 계산
            area_ratio_to_image = (area / total_image_area) * 100
            area_ratio_to_largest = (
                (area / largest_furniture_area) * 100
                if largest_furniture_area > 0
                else 0
            )

            # 크기 등급 결정
            if area_ratio_to_image >= 3.0:
                size_rank = "대형"
            elif area_ratio_to_image >= 1.0:
                size_rank = "중형"
            else:
                size_rank = "소형"

            # 상대 점수 (가장 큰 가구 = 100점)
            relative_size_score = round(area_ratio_to_largest, 1)

            size_info = {
                "name": name,
                "absolute_area": area,
                "area_ratio_to_image": f"{area_ratio_to_image:.2f}%",
                "area_ratio_to_largest": f"{area_ratio_to_largest:.1f}%",
                "size_rank": size_rank,
                "relative_size_score": relative_size_score,
            }

            furniture_size_comparison.append(size_info)
            print(
                f"  📏 {name:15s}: {area:,}px² ({area_ratio_to_image:.2f}%, {size_rank}, 점수: {relative_size_score})"
            )

        # 크기순으로 정렬
        furniture_size_comparison.sort(key=lambda x: x["absolute_area"], reverse=True)

        size_analysis = {
            "image_dimensions": {"width": img_width, "height": img_height},
            "total_image_area": total_image_area,
            "largest_furniture_area": largest_furniture_area,
            "furniture_count": len(furniture_list),
            "size_distribution": {
                "large": len(
                    [
                        f
                        for f in furniture_list
                        if (f.get("area", 0) / total_image_area) >= 0.03
                    ]
                ),
                "medium": len(
                    [
                        f
                        for f in furniture_list
                        if 0.01 <= (f.get("area", 0) / total_image_area) < 0.03
                    ]
                ),
                "small": len(
                    [
                        f
                        for f in furniture_list
                        if (f.get("area", 0) / total_image_area) < 0.01
                    ]
                ),
            },
            "furniture_size_comparison": furniture_size_comparison,
        }

        print(
            f"✅ 크기 분석 완료: 대형 {size_analysis['size_distribution']['large']}개, 중형 {size_analysis['size_distribution']['medium']}개, 소형 {size_analysis['size_distribution']['small']}개"
        )

        return size_analysis

    def crop_furniture_centered_filtered(
        self, image_path, furniture_list, output_dir="furniture_crops_filtered"
    ):
        """각 가구를 중심에 맞춰서 크롭 (필터링된 버전)"""
        # 출력 디렉토리 생성

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 이미지 열기
        img = open_image_by_type(image_path)
        img_width, img_height = img.size
        if isinstance(img, str):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = "IMG_FROM_RHINO"
        print(f"🎯 {len(furniture_list)}개 가구를 중심 맞춤 크롭합니다...")

        # 우선순위별로 정렬 (면적 기준 내림차순)
        sorted_furniture = sorted(
            furniture_list, key=lambda x: x.get("area", 0), reverse=True
        )

        cropped_images = []

        for i, furniture in enumerate(sorted_furniture):
            name = furniture["name"]
            category = furniture.get("category", "medium")
            priority = furniture.get("priority", "medium")
            x1, y1, x2, y2 = furniture["box"]
            area = furniture.get("area", 0)

            try:
                # 원본 가구 크기
                furniture_width = x2 - x1
                furniture_height = y2 - y1
                furniture_center_x = (x1 + x2) // 2
                furniture_center_y = (y1 + y2) // 2

                # 크롭 크기 결정 (카테고리별 여백 조정)
                if category == "large":
                    margin_ratio = 0.05  # 큰 가구는 5% 여백
                elif category == "medium":
                    margin_ratio = 0.1  # 중간 가구는 10% 여백
                else:
                    margin_ratio = 0.15  # 작은 가구는 15% 여백

                margin_x = int(furniture_width * margin_ratio)
                margin_y = int(furniture_height * margin_ratio)
                crop_width = furniture_width + 2 * margin_x
                crop_height = furniture_height + 2 * margin_y

                # 중심 맞춤 크롭 좌표 계산
                crop_x1 = max(0, furniture_center_x - crop_width // 2)
                crop_y1 = max(0, furniture_center_y - crop_height // 2)
                crop_x2 = min(img_width, crop_x1 + crop_width)
                crop_y2 = min(img_height, crop_y1 + crop_height)

                # 경계 조정
                if crop_x2 - crop_x1 < crop_width:
                    crop_x1 = max(0, crop_x2 - crop_width)
                if crop_y2 - crop_y1 < crop_height:
                    crop_y1 = max(0, crop_y2 - crop_height)

                # 이미지 크롭
                cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # 파일명 정리 (우선순위 포함)
                safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                filename = f"{base_name}_{i+1:02d}_{priority}_{safe_name}.png"
                filepath = os.path.join(output_dir, filename)

                # 저장
                cropped_img.save(filepath)

                # # 결과 정보 저장
                # crop_info = {
                #     "original_furniture": furniture,
                #     "crop_coordinates": [crop_x1, crop_y1, crop_x2, crop_y2],
                #     "crop_size": f"{crop_x2-crop_x1}x{crop_y2-crop_y1}",
                #     "filename": filename,
                #     "filepath": filepath,
                # }
                cropped_images.append(cropped_img)

                area_ratio = area / (img_width * img_height) * 100
                print(
                    f"✅ {i+1:2d}. {name:15s} ({priority:6s}) → {filename} ({crop_x2-crop_x1}x{crop_y2-crop_y1}, {area_ratio:.1f}%)"
                )

            except Exception as e:
                print(f"❌ {name} 크롭 실패: {e}")

        print(f"🎉 총 {len(cropped_images)}개 가구 크롭 완료!")
        print(f"📁 저장 위치: {output_dir}/")

        return cropped_images


class BackgroundRemover:
    # =============================================================================
    # 단계 2: Bria Remove Background 모델 사용
    # =============================================================================
    def __init__(self, replicate_client):
        self.replicate_client = replicate_client

    def process(
        self,
        cropped_files,
    ):
        """Bria 모델을 사용한 배경 제거 일괄 처리"""

        print("=" * 60)
        print(f"🎭 Bria 배경 제거 작업 시작")
        print(f"📊 처리할 파일 수: {len(cropped_files)}개")
        print(f"🤖 모델: bria/remove-background")
        print("=" * 60)

        processed_files = []
        success_count = 0

        for i, cropped_file in enumerate(cropped_files, 1):

            # 출력 파일명 생성

            print(f"\n🔄 [{i}/{len(cropped_files)}] 처리 중")

            # Bria 배경 제거 시도
            success, background_removed_file_byte = self.remove_background_per_file(
                cropped_file
            )

            if success:
                success_count += 1
                processed_files.append(background_removed_file_byte)

            else:
                # 실패한 경우 원본 파일 복사
                print(f"   ⚠️  배경 제거 실패")

            # API 호출 간 잠시 대기 (Rate Limit 방지)
            if i < len(cropped_files):
                print("   ⏳ 2초 대기 중...")
                time.sleep(2)

        print("\n" + "=" * 60)
        print(f"🎉 Bria 배경 제거 작업 완료!")
        print(f"✅ 성공: {success_count}/{len(cropped_files)}개")
        print("=" * 60)

        print(f"💾 결과 저장: step2_background_removal_bria.json")

        return processed_files

    def remove_background_per_file(self, cropped_image) -> Tuple[bool, bytes]:
        """Bria remove-background 모델을 사용하여 배경 제거"""
        print(f"🎭 Bria 배경 제거 시작: ")

        cropped_image = pil_to_filelike(cropped_image)
        # Bria 모델 실행
        output = self.replicate_client.run(
            "bria/remove-background",
            input={
                "image": cropped_image,
                "content_moderation": False,
                "preserve_partial_alpha": True,
            },
        )

        print(f"🔍 Bria 응답 타입: {type(output)}")

        # output을 메모리에서 읽기
        image_bytes = output.read()

        # PIL로 로드해서 남은 픽셀 수 확인
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        alpha = img.split()[-1]  # 알파 채널
        non_zero_pixels = sum(alpha.getdata()) // 255  # 0~255 값 → 픽셀 수로 변환

        print(f"   🟢 남은 픽셀 수: {non_zero_pixels}")

        if non_zero_pixels < 30:
            print(f"   ⚠️ 남은 픽셀이 {non_zero_pixels}개 → None 리턴")
            return False, None

        print(f"✅ Bria 배경 제거 완료")
        return True, image_bytes


class ImgToModeling:
    def __init__(self, replicate_client):
        self.replicate_client = replicate_client

    # =============================================================================
    # 단계 3: HunYuan3D 모델을 사용한 3D 변환
    # =============================================================================
    def run_hunyuan3d(self, image, output_filename):
        """HunYuan3D 모델을 사용하여 가구 이미지를 3D 모델로 변환"""
        try:
            print(f"🎨 3D 변환 시작:")

            input_data = {
                "image": image,
                "remove_background": False,  # 이미 배경이 제거됨
            }

            # HunYuan3D 모델 실행
            output = self.replicate_client.run(
                "ndreca/hunyuan3d-2:0602bae6db1ce420f2690339bf2feb47e18c0c722a1f02e9db9abd774abaff5d",
                input=input_data,
            )

            print(f"🔍 HunYuan3D 응답 타입: {type(output)}")

            # 3D 메시 다운로드
            mesh_url = output["mesh"]
            response = requests.get(mesh_url)

            with open(output_filename, "wb") as file:
                file.write(response.content)

            print(f"✅ 3D 모델 생성 완료: {output_filename}")
            return True

        except Exception as e:
            print(f"❌ 3D 변환 실패 : {e}")
            return False

    def process(self, selected_images, output_dir="furniture_3d_models"):
        """Bria 배경 제거된 가구들을 3D 모델로 변환"""
        try:
            # 출력 디렉토리 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("=" * 60)
            print(f"🎨 HunYuan3D 3D 변환 작업 시작")
            print(f"📁 출력 디렉토리: {output_dir}")
            print(f"📊 처리할 파일 수: {len(selected_images)}개")
            print(f"🤖 모델: ndreca/hunyuan3d-2")
            print("=" * 60)

            processed_files = []
            success_count = 0
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, selected_image in enumerate(selected_images, 1):

                output_filename = f"3d_{i}_{time_str}.glb"
                output_path = os.path.join(output_dir, output_filename)

                print(f"\n🔄 [{i}/{len(selected_images)}] 3D 변환 중:")

                # HunYuan3D 3D 변환 시도
                success = self.run_hunyuan3d(selected_image, output_path)

                if success:
                    success_count += 1

                    # 파일 크기 확인
                    model_size = (
                        os.path.getsize(output_path)
                        if os.path.exists(output_path)
                        else 0
                    )

                    file_info = {
                        "output_file": output_filename,
                        "output_path": output_path,
                        "model_size": model_size,
                        "model_used": "ndreca/hunyuan3d-2",
                        "status": "success",
                    }
                    processed_files.append(output_path)
                    print(f"   📊 3D 모델 크기: {model_size:,} bytes")
                    print(f"   💾 저장됨: {output_filename}")

                else:
                    file_info = {
                        "output_file": output_filename,
                        "output_path": output_path,
                        "model_used": "ndreca/hunyuan3d-2",
                        "status": "failed",
                    }
                    processed_files.append(file_info)
                    print(f"   ❌ 3D 변환 실패")

                # API 호출 간 잠시 대기 (Rate Limit 방지)
                if i < len(selected_images):
                    print("   ⏳ 5초 대기 중...")
                    time.sleep(5)  # 3D 변환은 더 많은 시간이 필요할 수 있음

            print("\n" + "=" * 60)
            print(f"🎉 3D 변환 작업 완료!")
            print(f"✅ 성공: {success_count}/{len(selected_images)}개")
            print(f"📁 결과 위치: {output_dir}/")
            print("=" * 60)

            return processed_files

        except Exception as e:
            print(f"❌ 3D 변환 작업 오류: {e}")
            return []
