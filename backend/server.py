"""
FastAPI server for building facade analysis.
Serves the React frontend and provides ML API endpoints.
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ml_pipeline import FacadeAnalyzer
from repair_calculator import RepairCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: Optional[FacadeAnalyzer] = None
calculator = RepairCalculator(total_area_m2=450.0)
results_store: dict = {}  # In-memory storage for analysis results

# Path to React build output
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup."""
    global analyzer
    logger.info("Starting facade analysis server...")
    analyzer = FacadeAnalyzer()
    analyzer.load_models()
    logger.info("Server ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Facade Analyzer API",
    description="ML-powered building facade analysis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ──

@app.get("/api/health")
async def health_check():
    """Check server status and model readiness."""
    import torch
    return {
        "status": "ok",
        "models_loaded": analyzer is not None and analyzer.models_loaded,
        "device": str(analyzer.device) if analyzer else "unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    facade_width: float = 20.0,
    facade_height: float = 12.0,
):
    """Upload a facade photo for analysis."""
    if analyzer is None or not analyzer.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(image_bytes) > 50_000_000:
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        logger.info(f"Analyzing image: {file.filename} ({len(image_bytes)} bytes)")

        total_area_m2 = round(facade_width * facade_height, 1)

        result = analyzer.analyze(image_bytes)
        analysis_id = result["id"]

        calc = RepairCalculator(total_area_m2=total_area_m2)
        repair_estimate = calc.calculate(
            damages=result["damages"],
            total_area_px=result["total_area_px"],
            layer_analysis=result.get("layer_analysis", {}),
        )

        results_store[analysis_id] = {"image_paths": result["image_paths"]}

        response = {
            "id": analysis_id,
            "overall_score": result["overall_score"],
            "overall_condition": result["overall_condition"],
            "facade_width": facade_width,
            "facade_height": facade_height,
            "total_area_m2": total_area_m2,
            "total_area_px": result["total_area_px"],
            "damaged_area_px": result["damaged_area_px"],
            "damaged_area_m2": round(
                (result["damaged_area_px"] / result["total_area_px"] * total_area_m2)
                if result["total_area_px"] > 0 else 0, 1
            ),
            "damages": result["damages"],
            "materials": result["materials"],
            "layer_analysis": {
                k: {
                    "area_px": v["area_px"],
                    "affected_layers": v["affected_layers"],
                    "crack_depth": v.get("crack_depth"),
                }
                for k, v in result.get("layer_analysis", {}).items()
            },
            "repair_estimate": repair_estimate,
            "processed_images": result["processed_images"],
        }

        logger.info(f"Analysis complete: score={result['overall_score']}, id={analysis_id}")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/results/{analysis_id}/images/{image_type}")
async def get_result_image(analysis_id: str, image_type: str):
    """Get a processed image from analysis results."""
    if analysis_id not in results_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    paths = results_store[analysis_id].get("image_paths", {})
    if image_type not in paths:
        raise HTTPException(
            status_code=404,
            detail=f"Image type '{image_type}' not found. Available: {list(paths.keys())}"
        )

    path = paths[image_type]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/results/{analysis_id}/images")
async def list_result_images(analysis_id: str):
    """List available processed images for an analysis."""
    if analysis_id not in results_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    paths = results_store[analysis_id].get("image_paths", {})
    return {"images": list(paths.keys())}


# ── Serve React Frontend ──

if os.path.isdir(STATIC_DIR):
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve React SPA — all non-API routes return index.html."""
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "name": "Facade Analyzer API",
            "version": "2.0.0",
            "status": "running",
            "note": "Frontend not built. Run: cd frontend && npm run build",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9000"))
    uvicorn.run(app, host=host, port=port)
