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
            d, s = cached_plan["decision"], cached_plan["sync"]
            logger.info(
                f"Returning cached transition plan: strategy={d['strategy']} score={d['score']} "
                f"confidence={d['confidence']} target_bpm={s['target_bpm']} "
                f"ratio_a={s['track_a_speed_ratio']} ratio_b={s['track_b_speed_ratio']}"
            )
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

        # Full detection detail, not just the final plan — this is what lets a listening-test
        # report ("phasing on this pair") be cross-referenced against what was actually detected,
        # since the mobile client has no access to these intermediate values at all.
        logger.info(
            f"Transition plan: strategy={plan.decision.strategy} score={plan.decision.score} "
            f"confidence={plan.decision.confidence} | "
            f"tempo_a={features_a.tempo:.2f} tempo_b={features_b.tempo:.2f} "
            f"target_bpm={plan.sync.target_bpm} ratio_a={plan.sync.track_a_speed_ratio} "
            f"ratio_b={plan.sync.track_b_speed_ratio} | "
            f"key_a={features_a.camelot_key}(conf={features_a.key_confidence:.2f}) "
            f"key_b={features_b.camelot_key}(conf={features_b.key_confidence:.2f}) | "
            f"downbeat_conf_a={features_a.downbeat_confidence:.2f} "
            f"downbeat_conf_b={features_b.downbeat_confidence:.2f} | "
            f"crossfade_execution_s={plan.timing.transition_duration_execution:.2f}"
        )
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
