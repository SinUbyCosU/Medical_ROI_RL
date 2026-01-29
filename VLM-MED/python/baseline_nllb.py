import time
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

TRANSLATOR_ID = "facebook/nllb-200-distilled-600M"
LLM_ID = "microsoft/Phi-3.5-mini-instruct"
TEST_PROMPTS: List[str] = [
    "Mera wifi nahi chal raha, light blink kar rahi hai.",
    "Laptop overheat ho raha hai.",
]


def robust_translate_pipeline(translator, llm, hinglish_prompt: str) -> str:
    eng_prompt = translator(
        hinglish_prompt,
        src_lang="hin_Deva",
        tgt_lang="eng_Latn",
    )[0]["translation_text"]

    eng_response = llm(
        eng_prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    )[0]["generated_text"]

    final_response = translator(
        eng_response,
        src_lang="eng_Latn",
        tgt_lang="hin_Deva",
    )[0]["translation_text"]

    return final_response


def main() -> None:
    translator = pipeline("translation", model=TRANSLATOR_ID, device=0)

    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_ID)
    llm = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device=0)

    for prompt in TEST_PROMPTS:
        start = time.perf_counter()
        response = robust_translate_pipeline(translator, llm, prompt)
        latency = time.perf_counter() - start
        print(f"Input: {prompt}")
        print(f"Output: {response}")
        print(f"Latency: {latency:.2f}s")
        print("-" * 20)


if __name__ == "__main__":
    main()
