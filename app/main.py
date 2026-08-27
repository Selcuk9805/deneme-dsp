from fastapi import FastAPI, HTTPException
from app.models.schemas import TransitionRequest, TransitionPlan
from app.services.downloader import DownloaderService
from app.services.analyzer import AnalyzerService
from app.services.transition import TransitionService

app = FastAPI(title="Automix DSP Backend")

@app.post("/api/transition/plan", response_model=TransitionPlan)
def get_transition_plan(request: TransitionRequest):
    try:
        path_a = DownloaderService.get_audio_path(request.track_a)
        path_b = DownloaderService.get_audio_path(request.track_b)
        
        features_a = AnalyzerService.analyze_track(path_a, is_track_a=True)
        features_b = AnalyzerService.analyze_track(path_b, is_track_a=False)
        
        plan = TransitionService.calculate_transition(features_a, features_b)
        
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ok"}
