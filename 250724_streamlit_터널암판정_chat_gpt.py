import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dotenv import load_dotenv
import base64
import httpx
from openai import OpenAI
import urllib3
import tempfile
import gc

# 환경 변수 로드 및 OpenAI 클라이언트 설정
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)

# 모델 로드
@st.cache_resource
def load_model():
    model = torch.load("./model_convnext_tiny2.pth", weights_only = False, map_location=torch.device('cpu') )
    model.eval()
    return model

# 이미지 전처리
def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# PIL → base64
def pil_to_base64(image: Image.Image) -> str:
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Grad-CAM 계산
def generate_gradcam(model, input_tensor, original_np):
    target_layer = model.features[-1][-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
    return cam_image

# GPT-5-mini 분석
def analyze_with_gpt4o(original_img: Image.Image, cam_img: Image.Image, label_idx_: int, class_name: str) -> str:
    original_base64 = pil_to_base64(original_img)
    cam_base64 = pil_to_base64(cam_img)

    prompt = (
        f"이 이미지는 RMR 암반 분류 중 Class {label_idx_} ({class_name})로 예측되었습니다.\n"
        "두 이미지를 참고하여 Grad-CAM 결과를 반영한 기술적 분석을 수행하십시오.\n"
        "1. 암반 표면 풍화 상태\n"
        "2. 절리 간격 및 방향성\n"
        "3. 암석 강도\n"
        "4. 구조적 안정성\n"
        "5. Grad-CAM 강조 영역의 지질학적 의미\n"
        "필요하시면 또는 원하신다면 어떤것을 작성해 드리겠습니다는 제외하고 답변"
    )

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "당신은 지질공학 및 암반공학 전문가입니다."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cam_base64}"}}
                ]
            }
        ],
        temperature=1
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("터널 안전성 AI🤖 분류 서비스")

uploaded_file = st.file_uploader("터널 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 임시 파일 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    image_path = temp_file.name

    # 이미지 로딩
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="업로드된 원본 이미지", use_container_width=True)

    # 모델 불러오기
    with st.spinner("모델 로딩 및 예측 중..."):
        model = load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = preprocess(image).to(device)
        original_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

        # 예측
        pred = torch.argmax(model(input_tensor), dim=1).item() + 1
        rmr_classes = {
            1: "Very Good Rock",
            2: "Good Rock",
            3: "Fair Rock",
            4: "Poor Rock",
            5: "Very Poor Rock"
        }
        rmr_class_name = rmr_classes.get(pred, "Unknown")

        # Grad-CAM 생성
        cam_image = generate_gradcam(model, input_tensor, original_np)

    # 시각화
    st.subheader(f"📊 예측 결과: RMR Class {pred} ({rmr_class_name})")
    col1, col2 = st.columns(2)
    col1.image(image.resize((336, 336)), caption="원본 이미지", use_container_width=True)
    col2.image(cam_image, caption="Grad-CAM 시각화", use_container_width=True)

    # GPT-5-mini 분석 실행
    with st.spinner("gpt5-mini 분석 중..."):
        cam_pil = Image.fromarray(cam_image).resize((336, 336))
        original_resized = image.resize((336, 336))
        result = analyze_with_gpt4o(original_resized, cam_pil, pred, rmr_class_name)
        st.success("✅ 분석 완료")

    st.subheader("🧠 GPT5-mini 기반 기술 분석")
    st.markdown(result)

    # 메모리 해제
    gc.collect()






