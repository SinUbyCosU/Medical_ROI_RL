import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

prompts = {
    "Hinglish (Standard)": "Windows mein DNS flush kaise karte hain?",
    "Transliterated (Baseline)": "विंडोज में डीएनएस फ्लश कैसे करते हैं?",
}

print("Running Transliteration Baseline...")

for label, text in prompts.items():
    msgs = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        msgs,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=100, do_sample=False)

    response = tokenizer.decode(output[0]).split("<|assistant|>")[-1].strip()
    print(f"\n[{label} Input]: {text}")
    print(f"[Response]: {response}")
