import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class CLAS_Model:
    def get_hidden_states(self, prompt, layer=16):
        """
        Returns the mean hidden states for the given prompt at the specified layer.
        """
        messages = [
            {"role": "system", "content": "You are a helpful technical assistant. Answer the user's query in natural, colloquial Hinglish."},
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.model.device)
        activations = {}
        def hook(module, input, output):
            activations["hidden"] = output[0].detach()
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            self.model(**inputs)
        handle.remove()
        return activations["hidden"]

    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "microsoft/Phi-3.5-mini-instruct"
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.config.use_cache = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt, injection_layer=16, alpha=0.05):
        # Prepare prompt
        messages = [
            {"role": "system", "content": "You are a helpful technical assistant. Answer the user's query in natural, colloquial Hinglish."},
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.model.device)

        # Get English and Hinglish activations (simulate with prompt for both)
        # For sweep, we use only the Hinglish prompt, so we just steer with a fixed vector (simulate)
        # In a real experiment, you would precompute the English vector and subtract the Hinglish vector
        # Here, we use a dummy zero vector for simplicity
        with torch.no_grad():
            # Get baseline activation
            activations = {}
            def hook(module, input, output):
                activations["hin"] = output[0].detach()
            handle = self.model.model.layers[injection_layer].register_forward_hook(hook)
            self.model(**inputs)
            handle.remove()
            hin_vec = activations["hin"].mean(dim=1, keepdim=True)

        # For demo, use a random vector as the "steering" direction
        steering_vec = torch.randn_like(hin_vec) * 0.01

        def steering_hook(module, input, output):
            current = output[0]
            if steering_vec.shape[1] == 1 and current.shape[1] > 1:
                vec = steering_vec.expand(current.shape[0], current.shape[1], steering_vec.shape[-1])
            else:
                vec = steering_vec
            new_hidden = current + alpha * vec
            return (new_hidden,) + output[1:]

        # Register steering hook
        handle = self.model.model.layers[injection_layer].register_forward_hook(steering_hook)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.6,
                use_cache=False
            )
        handle.remove()
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return response.split('assistant')[-1].strip()
