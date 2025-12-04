import logging
import os
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Optional
from openai import OpenAI

# Configuration
SOLAR_API_KEY = os.getenv("SOLAR_API_KEY", "")  # Upstage Solar API Key (환경변수로 설정 필요)
BACKEND_URL = "http://3.37.253.134:8080/api/polls"
BACKEND_AUTH_URL = "http://3.37.253.134:8080/api/auth"
AI_ADMIN_USERNAME = "AI_Admin"
AI_ADMIN_PASSWORD = "AIAdmin2025!SecurePass"  # 강력한 비밀번호
SCHEDULE_INTERVAL_SEC = 3600  # 1시간마다 (3600초)

# Initialize Solar API client (will be set during startup)
solar_client = None

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("JJiGiT-AI")

app = FastAPI(title="JJiGiT AI Service")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://jjigit-fe.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication Manager
class AuthManager:
    """Manages JWT token authentication with backend."""

    def __init__(self):
        self.token: Optional[str] = None
        self.user_id: Optional[int] = None

    def signup(self) -> bool:
        """Create AI admin account."""
        try:
            payload = {
                "username": AI_ADMIN_USERNAME,
                "password": AI_ADMIN_PASSWORD
            }
            response = requests.post(f"{BACKEND_AUTH_URL}/signup", json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                self.user_id = data.get("userId")
                logger.info(f"AI Admin account created: userId={self.user_id}")
                return True
            elif response.status_code == 409:
                # Account already exists
                logger.info("AI Admin account already exists.")
                return True
            else:
                logger.error(f"Signup failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Signup error: {e}")
            return False

    def login(self) -> bool:
        """Login and obtain JWT token."""
        try:
            payload = {
                "username": AI_ADMIN_USERNAME,
                "password": AI_ADMIN_PASSWORD
            }
            response = requests.post(f"{BACKEND_AUTH_URL}/login", json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                self.token = data.get("token")
                self.user_id = data.get("userId")
                logger.info(f"Login successful: userId={self.user_id}")
                return True
            else:
                logger.error(f"Login failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, create account and login if needed."""
        # Try to login first
        if self.login():
            return True

        # If login fails, try to signup then login
        logger.info("Login failed, attempting to create account...")
        if self.signup() and self.login():
            return True

        logger.error("Failed to authenticate AI service.")
        return False

    def get_auth_headers(self) -> dict:
        """Get authorization headers for API requests."""
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

# Initialize auth manager
auth_manager = AuthManager()

def generate_discussion_topic_solar():
    """Generates a discussion topic using Upstage Solar API."""
    if not solar_client:
        logger.error("Solar API client is not initialized")
        return None

    try:
        response = solar_client.chat.completions.create(
            model="solar-pro2",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 흥미로운 찬반 투표 주제를 생성하는 AI입니다. 간결하고 명확하며 논쟁적인 주제를 한 문장으로 제안하세요."
                },
                {
                    "role": "user",
                    "content": "사람들이 찬성과 반대로 의견을 나눌 수 있는 흥미로운 토론 주제를 한 개만 생성해주세요. 주제만 간결하게 작성하고, 설명이나 부가 내용은 포함하지 마세요."
                }
            ],
            temperature=0.8,
            max_tokens=100,
            stream=False
        )

        topic = response.choices[0].message.content.strip()

        # 따옴표나 불필요한 문장부호 제거
        topic = topic.strip('"\'').strip()

        logger.info(f"Solar API generated topic: {topic}")
        return topic

    except Exception as e:
        logger.error(f"Error calling Solar API: {e}")
        return None

def generate_discussion_topic():
    """Generates a discussion topic using Solar API."""
    return generate_discussion_topic_solar()

def generate_poll_data():
    """Generates a complete poll with topic and options."""
    # Generate topic
    topic = generate_discussion_topic()

    if not topic:
        logger.error("Failed to generate topic")
        return None

    # Create poll structure with options
    # For debate topics, we create "찬성" (Agree) and "반대" (Disagree) options
    poll_data = {
        "title": topic,
        "isPublic": True,
        "options": [
            {
                "optionText": "찬성",
                "optionOrder": 1
            },
            {
                "optionText": "반대",
                "optionOrder": 2
            }
        ]
    }

    return poll_data

def upload_poll_to_backend(poll_data: dict) -> bool:
    """Upload poll to backend API."""
    try:
        # Ensure we have authentication
        if not auth_manager.token:
            logger.warning("No auth token, attempting to authenticate...")
            if not auth_manager.ensure_authenticated():
                logger.error("Cannot upload poll: Authentication failed")
                return False

        # Get auth headers
        headers = auth_manager.get_auth_headers()
        headers["Content-Type"] = "application/json"

        # Send request to backend
        response = requests.post(BACKEND_URL, json=poll_data, headers=headers, timeout=10)

        if response.status_code == 200:
            result = response.json()
            poll_id = result.get("pollId")
            logger.info(f"Successfully uploaded poll to backend: pollId={poll_id}")
            return True
        elif response.status_code == 401:
            # Token expired, try to re-authenticate
            logger.warning("Token expired, re-authenticating...")
            if auth_manager.ensure_authenticated():
                # Retry with new token
                headers = auth_manager.get_auth_headers()
                headers["Content-Type"] = "application/json"
                response = requests.post(BACKEND_URL, json=poll_data, headers=headers, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    poll_id = result.get("pollId")
                    logger.info(f"Successfully uploaded poll after re-auth: pollId={poll_id}")
                    return True
            logger.error(f"Backend upload failed after re-auth: {response.status_code}")
            return False
        else:
            logger.error(f"Backend upload failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error uploading poll to backend: {e}")
        return False

def job_auto_create_topic():
    """Scheduled job to generate and upload a new poll."""
    logger.info("Starting scheduled poll generation...")

    # Generate poll data (topic + options)
    poll_data = generate_poll_data()

    if poll_data:
        logger.info(f"Generated Poll: {poll_data['title']}")

        # Upload to backend
        if upload_poll_to_backend(poll_data):
            logger.info("Poll successfully created and uploaded!")
        else:
            logger.error("Failed to upload poll to backend")
    else:
        logger.error("Failed to generate poll data")

# Scheduler Setup
scheduler = BackgroundScheduler()
scheduler.add_job(job_auto_create_topic, 'interval', seconds=SCHEDULE_INTERVAL_SEC)
scheduler.start()

# API Schemas
class TopicResponse(BaseModel):
    topic: str

class PollResponse(BaseModel):
    title: str
    options: list

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize authentication and Solar API client on startup."""
    global solar_client

    logger.info("JJiGiT AI Service starting up...")

    # Initialize Solar API client
    if SOLAR_API_KEY:
        try:
            import httpx
            # Create httpx client without proxy settings to avoid compatibility issues
            http_client = httpx.Client(timeout=30.0)
            solar_client = OpenAI(
                api_key=SOLAR_API_KEY,
                base_url="https://api.upstage.ai/v1",
                http_client=http_client
            )
            logger.info("Solar API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Solar API client: {e}")
            raise e
    else:
        logger.error("SOLAR_API_KEY is not set. Please set the environment variable.")
        raise ValueError("SOLAR_API_KEY is required")

    # Initialize backend authentication
    if auth_manager.ensure_authenticated():
        logger.info("Authentication successful - AI service ready!")
    else:
        logger.warning("Authentication failed - poll upload will not work until backend is available")

# Endpoints
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "solar-pro2",
        "authenticated": auth_manager.token is not None,
        "user_id": auth_manager.user_id
    }

@app.post("/api/generate", response_model=TopicResponse)
def api_generate_topic():
    """Generate a discussion topic only."""
    topic = generate_discussion_topic()
    if not topic:
        return {"topic": "주제 생성에 실패했습니다."}
    return {"topic": topic}

@app.post("/api/generate-poll", response_model=PollResponse)
def api_generate_poll():
    """Generate a complete poll with topic and options."""
    poll_data = generate_poll_data()
    if not poll_data:
        return {"title": "주제 생성에 실패했습니다.", "options": []}
    return poll_data

@app.post("/api/create-poll")
def api_create_poll():
    """Generate and upload a poll to backend."""
    poll_data = generate_poll_data()
    if not poll_data:
        return {"success": False, "message": "주제 생성에 실패했습니다."}

    if upload_poll_to_backend(poll_data):
        return {"success": True, "message": "Poll created successfully", "poll": poll_data}
    else:
        return {"success": False, "message": "Failed to upload poll to backend"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)