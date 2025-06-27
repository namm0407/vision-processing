#runs fine

from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# First try with AutoModel
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-VL-1.8B-Instruct",  # Smaller model
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
except:
    # If that fails, use the specific model class
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)

print("Model and processor loaded successfully!")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "dogs.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
