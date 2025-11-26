import logging
import torch
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from apscheduler.schedulers.background import BackgroundScheduler

# Configuration
MODEL_NAME = "skt/kogpt2-base-v2"
BACKEND_URL = "http://localhost:8080/api/polls"  # 추후 백엔드 수정 필요
SCHEDULE_INTERVAL_SEC = 30  # 테스트용 30초

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("JJiGiT-AI")

app = FastAPI(title="JJiGiT AI Service")

# Load Model & Tokenizer
try:
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        MODEL_NAME, 
        bos_token='</s>', 
        es_token='</s>', 
        pad_token='<pad>'
    )
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

def generate_discussion_topic():
    """Generates a discussion topic using KoGPT2."""
    prompt = "한국 사회에서 20대가 가장 관심 있어 하는 찬반 토론 주제는"
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_length=50,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.8
            )
            
        generated_text = tokenizer.decode(gen_ids[0])
        clean_text = generated_text.replace("</s>", "").strip()
        return clean_text
        
    except Exception as e:
        logger.error(f"Error generating topic: {e}")
        return None

def job_auto_create_topic():
    """Scheduled job to generate and upload a new topic."""
    topic = generate_discussion_topic()
    
    if topic:
        logger.info(f"Scheduled Topic Generated: {topic}")
        # TODO: Enable this block after backend API integration
        # try:
        #     payload = {"title": topic, "author": "AI_Admin"}
        #     res = requests.post(BACKEND_URL, json=payload)
        #     if res.status_code == 200:
        #         logger.info("Successfully uploaded to backend.")
        #     else:
        #         logger.warning(f"Backend upload failed: {res.status_code}")
        # except Exception as req_err:
        #     logger.error(f"Request failed: {req_err}")

# Scheduler Setup
scheduler = BackgroundScheduler()
scheduler.add_job(job_auto_create_topic, 'interval', seconds=SCHEDULE_INTERVAL_SEC)
scheduler.start()

# API Schemas
class TopicResponse(BaseModel):
    topic: str

# Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/api/generate", response_model=TopicResponse)
def api_generate_topic():
    topic = generate_discussion_topic()
    if not topic:
        return {"topic": "주제 생성에 실패했습니다."}
    return {"topic": topic}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)