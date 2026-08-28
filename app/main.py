import logging
from fastapi import FastAPI, HTTPException
from app.models.schemas import TransitionRequest, TransitionPlan
from app.services.downloader import DownloaderService
from app.services.analyzer import AnalyzerService
from app.services.transition import TransitionService
from app.services.database import DatabaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Automix DSP Backend")

@app.on_event("startup")
def startup_event():
    logger.info("Starting up Automix DSP Backend...")
    DownloaderService.init_cache()
    DatabaseService.init_db()

@app.post("/api/transition/plan", response_model=TransitionPlan)
def get_transition_plan(request: TransitionRequest):
    try:
        logger.info(f"Received request for transition: {request.track_a} -> {request.track_b}")
        
        # 1. Check Database Cache
        cached_plan = DatabaseService.get_plan(request.track_a, request.track_b)
        if cached_plan:
            logger.info("Returning cached transition plan.")
            return TransitionPlan(**cached_plan)
            
        # 2. Cache Miss: Download and Analyze
        logger.info("Calculating new transition plan...")
        path_a = DownloaderService.get_audio_path(request.track_a)
        path_b = DownloaderService.get_audio_path(request.track_b)
        
        features_a = AnalyzerService.analyze_track(path_a, is_track_a=True)
        features_b = AnalyzerService.analyze_track(path_b, is_track_a=False)
        
        # 3. Calculate Transition
        plan = TransitionService.calculate_transition(features_a, features_b)
        
        # 4. Save to Database
        DatabaseService.save_plan(request.track_a, request.track_b, plan.dict())
        
        logger.info("Transition plan generated and cached successfully.")
        return plan
    except Exception as e:
        logger.error(f"Error generating transition plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/clear")
def clear_cache():
    try:
        logger.info("Received request to clear cache.")
        DownloaderService.clear_cache()
        DatabaseService.clear_cache()
        return {"status": "success", "message": "Audio and Database caches cleared successfully."}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ok"}
