from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"

def load_model():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    text_gen=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.6,
        top_p=0.9,
    )
    return text_gen

generator=load_model()

def generate_topic():
    prompt="""
    한국 사회나 기술 관련 토론 주제 하나를 간단하고 중립적으로 생성해 주세요.
주제는 한 문장으로, 현재 이슈에 맞는 내용을 다뤄야 합니다."""
    
    response=generator(prompt)[0]["generated_text"]
    
    cleaned=response.strip().split("\n")[-1]
    
    return cleaned

if __name__=="__main__":
    print("=== Generated Topic ===")
    print(generate_topic())