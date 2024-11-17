import os
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from logger import setup_logger

logger = setup_logger(log_file_prefix="inference")

# 이미지 불러오기, 전처리
def load_and_preprocess_images(folder_path, transform):
    image_paths = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    processed_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("L")
        processed_image = transform(image)
        processed_images.append(processed_image.numpy())
    
    return np.stack(processed_images).astype(np.float32), image_paths

# InferInput 객체 생성
def create_infer_input(batch, input_name="INPUT__0"):
    infer_input = InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    return infer_input

# 추론
def perform_inference(client, model_name, infer_input, output_name="OUTPUT__0"):
    infer_output = InferRequestedOutput(output_name)
    start_time = time.time()
    response = client.infer(model_name=model_name, inputs=[infer_input], outputs=[infer_output])
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # ms
    return response.as_numpy(output_name), latency

# 로그 기록
def log_results(image_paths, results, latency):
    # Inference Latency
    logger.info(f"Inference Latency: {latency:.2f} ms")
    
    # 이미지 예측 결과
    predictions = []
    for i, result in enumerate(results):
        predicted_class = np.argmax(result)
        predictions.append(f"Image {i+1}, Predicted class: {predicted_class}, Image file: {image_paths[i]}")
    
    logger.info("Image Predictions:\n" + "\n".join(predictions))

def main():
    try:
        triton_client = InferenceServerClient(url="triton-server-service:8001")
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {e}")
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_batch, image_paths = load_and_preprocess_images("/app/mnist/inference", transform)
    
    infer_input = create_infer_input(input_batch)
    
    results, latency = perform_inference(triton_client, "resnet18_mnist", infer_input)
    
    log_results(image_paths, results, latency)

if __name__ == "__main__":
    main()
