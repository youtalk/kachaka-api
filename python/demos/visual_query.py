import requests
from PIL import Image
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor

import io
import grpc
import kachaka_api_pb2 as pb2
from kachaka_api_pb2_grpc import KachakaApiStub

stub = KachakaApiStub(grpc.insecure_channel("172.18.209.213:26400"))

model = AutoModelForVision2Seq.from_pretrained(
    "stabilityai/japanese-instructblip-alpha", 
    trust_remote_code=True   
).to("cuda")
processor = BlipImageProcessor.from_pretrained(
    "stabilityai/japanese-instructblip-alpha"
)
tokenizer = LlamaTokenizer.from_pretrained(
    "novelai/nerdstash-tokenizer-v1", 
    additional_special_tokens=['▁▁']
)

prompt = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
与えられた画像について、詳細に述べてください。

### 応答:"""

def get_image():
    request = pb2.GetRequest()
    response = stub.GetFrontCameraRosCompressedImage(request)
    return Image.open(io.BytesIO(response.image.data)).convert("RGB")


def speak(text: str):
    command = pb2.Command(speak_command=pb2.SpeakCommand(text=text))
    request = pb2.StartCommandRequest(command=command, cancel_all=True)
    response = stub.StartCommand(request)
    return response.result.success
    

while True:
    image = get_image()
    print(image)
    inputs = processor(images=image, return_tensors="pt")
    text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
    text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
    inputs.update(text_encoding)
    outputs = model.generate(
        **inputs.to("cuda", dtype=model.dtype),
        num_beams=5,
        max_new_tokens=32,
        min_length=1,
    )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)
    speak(generated_text)
