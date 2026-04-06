import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import os
import uuid

app = FastAPI(title="Face Tracking Emoji API", description="A simple MLOps API to track faces in a video and overlay an emoji.")

# MediaPipe 얼굴 감지 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 디렉토리 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
EMOJI_PATH = os.path.join(ASSETS_DIR, "emoji.png")

# 필요한 디렉토리가 없으면 생성합니다.
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """지정된 좌표에 알파 채널을 적용하여 이미지를 오버레이합니다."""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

def process_video(input_path: str, output_path: str):
    # 이모지 이미지 로드 (없으면 노란색 원형으로 대체)
    if not os.path.exists(EMOJI_PATH):
        emoji_img = np.zeros((200, 200, 4), dtype=np.uint8)
        # 노란색 얼굴 기본 배경 (BGR + Alpha)
        cv2.circle(emoji_img, (100, 100), 95, (0, 255, 255, 255), -1)
        # 눈
        cv2.circle(emoji_img, (60, 70), 15, (0, 0, 0, 255), -1)
        cv2.circle(emoji_img, (140, 70), 15, (0, 0, 0, 255), -1)
        # 입
        cv2.ellipse(emoji_img, (100, 120), (50, 40), 0, 0, 180, (0, 0, 0, 255), 10)
    else:
        emoji_img = cv2.imread(EMOJI_PATH, cv2.IMREAD_UNCHANGED)
        
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 웹 브라우저 호환성을 위해 avc1 (H.264) 사용 시도 (시스템에 따라 mp4v를 사용해야 할 수도 있음)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe는 RGB 포맷을 요구합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                # 얼굴 바운딩 박스 계산
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # 자연스러운 오버레이를 위해 이모지 크기를 조정합니다.
                # 얼굴 크기보다 살짝 크게 설정
                emoji_size = int(max(w, h) * 1.3)
                
                # 오버레이 위치 조정 (센터를 맞추기 위함)
                adj_x = x - int((emoji_size - w) / 2)
                adj_y = y - int((emoji_size - h) / 2)

                if emoji_size > 0:
                    resized_emoji = cv2.resize(emoji_img, (emoji_size, emoji_size))
                    
                    if resized_emoji.shape[2] == 4:
                        # 알파 채널과 BGR 추출
                        alpha_mask = resized_emoji[:, :, 3] / 255.0
                        overlay_colors = resized_emoji[:, :, :3]
                        
                        # 프레임에 오버레이 적용
                        overlay_image_alpha(frame, overlay_colors, adj_x, adj_y, alpha_mask)

        out.write(frame)

    cap.release()
    out.release()

def cleanup_files(file_paths):
    """작업이 끝난 후 임시 파일들을 삭제합니다."""
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error removing file {path}: {e}")

@app.post("/api/v1/process-video/")
async def upload_and_process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """동영상 파일을 업로드받아 얼굴에 이모지를 합성한 후 반환합니다."""
    task_id = str(uuid.uuid4())
    input_filename = f"{task_id}_{file.filename}"
    output_filename = f"{task_id}_processed.mp4"
    
    input_path = os.path.join(TEMP_DIR, input_filename)
    output_path = os.path.join(TEMP_DIR, output_filename)

    # 업로드된 파일 저장
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 동영상 처리 수행
    process_video(input_path, output_path)

    # 응답 후 삭제되도록 백그라운드 태스크 등록
    background_tasks.add_task(cleanup_files, [input_path, output_path])

    # 처리된 동영상 반환
    return FileResponse(output_path, media_type="video/mp4", filename=f"emoji_{file.filename}")

@app.get("/")
def read_root():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
