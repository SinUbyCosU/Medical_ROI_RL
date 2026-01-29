import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
LAYER_IDX = 16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
vector = torch.load("vectors/tech_vector.pt").to(model.device)

prompt = "Mera wifi connect nahi ho raha, router reset kaise karun?"
configs = ["Normalized (CLAS)", "Unnormalized (Raw Addition)"]

print(f"Prompt: {prompt}\n" + "-" * 30)

for config_name in configs:

    def hook(module, args, output):
        if isinstance(output, tuple):
            hidden = output[0]
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        else:
            hidden = output

        if hidden.shape[1] > 1:
            if config_name == "Normalized (CLAS)":
                hidden_norm = hidden.norm(dim=-1, keepdim=True)
                v_unit = vector / vector.norm()
                hidden = hidden + v_unit * hidden_norm
            else:
                hidden = hidden + vector

        if isinstance(output, tuple):
            rest = tuple(output[1:]) if len(output) > 1 else ()
            return (hidden,) + rest
        if hasattr(output, "_replace"):
            return output._replace(last_hidden_state=hidden)
        return hidden

    handle = model.model.layers[LAYER_IDX].register_forward_hook(hook)

    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=80, do_sample=False)

    response = tokenizer.decode(output[0]).split("<|assistant|>")[-1].strip()
    print(f"\n[{config_name}]:\n{response}")

    handle.remove()
