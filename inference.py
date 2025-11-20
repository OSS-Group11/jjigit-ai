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
    당신은 토론 주제를 한 문장으로 생성하는 역할입니다.

규칙:
- 한국 사회, 문화, 기술과 관련된 주제 하나만 생성하세요.
- 번호나 불필요한 문장은 절대 포함하지 마세요.
- 문장은 자연스러운 한국어로 작성하세요.
- 부적절한 내용은 포함하지 마세요.

출력 형식:
<주제만 한 문장으로 출력>

지금 생성하세요:"""
    
    response=generator(prompt)[0]["generated_text"]
    
    cleaned=response.strip().split("\n")[-1]
    
    return cleaned

if __name__=="__main__":
    print("=== Generated Topic ===")
    print(generate_topic())