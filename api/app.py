"""
author: Jayesh Pandey
summary: FastAPI application for VLMJudge, providing endpoints for scoring, comparison, and feedback.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.runtime import InferenceRuntime

logger = logging.getLogger("imagereward_api")
_LOG_LOCK = threading.Lock()


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    imageA: str = Field(..., min_length=1)
    imageB: str = Field(..., min_length=1)


class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image: str = Field(..., min_length=1)


class BatchCompareItem(BaseModel):
    prompt: str = Field(..., min_length=1)
    imageA: str = Field(..., min_length=1)
    imageB: str = Field(..., min_length=1)

class ExplainRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    imageA: str = Field(..., min_length=1)
    imageB: str = Field(..., min_length=1)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class FeedbackRequest(BaseModel):
    correct_winner: str = Field(..., pattern="^(A|B|tie)$")
    prompt: str = Field(..., min_length=1)
    imageA: str = Field(..., min_length=1)
    imageB: str = Field(..., min_length=1)


def create_app(*, config_path: str = "config.yaml") -> FastAPI:
    runtime_holder: Dict[str, Any] = {"runtime": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        runtime_holder["runtime"] = InferenceRuntime.from_yaml(config_path)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("event=startup ms=%.2f device=%s", dt_ms, runtime_holder["runtime"].device)
        yield

    app = FastAPI(title="VLMJudge API", version="1.0.0", lifespan=lifespan)

    cors_env = os.getenv("VLMJUDGE_CORS_ORIGINS", "").strip()
    if cors_env:
        origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    else:
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            resp = await call_next(request)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=request method=%s path=%s ms=%.2f", request.method, request.url.path, dt_ms)
        return resp

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.warning("event=error path=%s err=%s", request.url.path, exc)
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/compare")
    async def compare(req: CompareRequest):
        rt: InferenceRuntime = runtime_holder["runtime"]
        if rt is None:
            raise HTTPException(status_code=503, detail="Service initializing. Please retry after startup completes.")
        try:
            out = rt.compare(prompt=req.prompt, image_a_ref=req.imageA, image_b_ref=req.imageB, return_debug=True)
            dbg = out.pop("_debug", {}) if isinstance(out, dict) else {}
            logger.info(
                "event=compare method=%s conf=%.3f ms_total=%.2f",
                out.get("method", ""),
                float(out.get("confidence", 0.0)),
                float(out.get("timing_ms", {}).get("total", 0.0)),
            )
            _append_jsonl(
                os.path.join("logs", "requests.jsonl"),
                {
                    "timestamp": time.time(),
                    "type": "compare",
                    "prompt": req.prompt,
                    "imageA": req.imageA,
                    "imageB": req.imageB,
                    "winner": out.get("winner", "tie"),
                    "confidence": float(out.get("confidence", 0.0)),
                    "method": out.get("method", "student"),
                    "latency_ms": float(out.get("timing_ms", {}).get("total", 0.0)),
                    "agreement": dbg.get("agreement", None),
                    "student_winner": dbg.get("student_winner", None),
                    "teacher_winner": dbg.get("teacher_winner", None),
                    "student_variant": dbg.get("student_variant", None),
                    "student_checkpoint": dbg.get("student_checkpoint", None),
                },
            )
            return out
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/score")
    async def score(req: ScoreRequest):
        rt: InferenceRuntime = runtime_holder["runtime"]
        if rt is None:
            raise HTTPException(status_code=503, detail="Service initializing. Please retry after startup completes.")
        try:
            out = rt.score(prompt=req.prompt, image_ref=req.image)
            dbg = out.pop("_debug", {}) if isinstance(out, dict) else {}
            _append_jsonl(
                os.path.join("logs", "requests.jsonl"),
                {
                    "timestamp": time.time(),
                    "type": "score",
                    "prompt": req.prompt,
                    "image": req.image,
                    "score": float(out.get("score", 0.5)),
                    "confidence": float(out.get("confidence", 0.0)),
                    "latency_ms": float(out.get("timing_ms", 0.0)),
                    "method": "student",
                    "student_variant": dbg.get("student_variant", None),
                },
            )
            return {"score": out.get("score", 0.5), "confidence": out.get("confidence", 0.0)}
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/batch_compare")
    async def batch_compare(req: List[BatchCompareItem]):
        rt: InferenceRuntime = runtime_holder["runtime"]
        if rt is None:
            raise HTTPException(status_code=503, detail="Service initializing. Please retry after startup completes.")
        outs = []
        for i, item in enumerate(req):
            try:
                out = rt.compare(prompt=item.prompt, image_a_ref=item.imageA, image_b_ref=item.imageB, return_debug=True)
                dbg = out.pop("_debug", {}) if isinstance(out, dict) else {}
                _append_jsonl(
                    os.path.join("logs", "requests.jsonl"),
                    {
                        "timestamp": time.time(),
                        "type": "batch_compare",
                        "idx": i,
                        "prompt": item.prompt,
                        "imageA": item.imageA,
                        "imageB": item.imageB,
                        "winner": out.get("winner", "tie"),
                        "confidence": float(out.get("confidence", 0.0)),
                        "method": out.get("method", "student"),
                        "latency_ms": float(out.get("timing_ms", {}).get("total", 0.0)),
                        "agreement": dbg.get("agreement", None),
                        "student_winner": dbg.get("student_winner", None),
                        "teacher_winner": dbg.get("teacher_winner", None),
                        "student_variant": dbg.get("student_variant", None),
                        "student_checkpoint": dbg.get("student_checkpoint", None),
                    },
                )
                outs.append(out)
            except Exception as e:
                outs.append({"error": str(e), "idx": i})
        return outs

    @app.post("/explain")
    async def explain(req: ExplainRequest):
        rt: InferenceRuntime = runtime_holder["runtime"]
        if rt is None:
            raise HTTPException(status_code=503, detail="Service initializing. Please retry after startup completes.")
        try:
            # Bypass scoring pipeline and directly use VLM ensemble if available.
            # If we don't expose VLM directly from runtime, we can just run the pipeline
            # and extract reasoning. The pipeline uses VLM cache anyway.
            out = rt.compare(prompt=req.prompt, image_a_ref=req.imageA, image_b_ref=req.imageB, return_debug=True)
            
            vlm_data = out.get("scores", {}).get("vlm", {})
            reasoning_short = vlm_data.get("reasoning_short", "")
            reasoning_full = vlm_data.get("reason", "")
            confidence = float(out.get("confidence", 0.0))
            
            if not reasoning_full:
                reasoning_full = out.get("reasoning", "")
                from vlmjudge.vlm.reasoning_utils import compress_reasoning
                reasoning_short = compress_reasoning(reasoning_full)

            return {
                "reasoning_short": reasoning_short,
                "reasoning_full": reasoning_full,
                "confidence": confidence
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/feedback")
    async def feedback(req: FeedbackRequest):
        rt: InferenceRuntime = runtime_holder["runtime"]
        if rt is None:
            raise HTTPException(status_code=503, detail="Service initializing. Please retry after startup completes.")
        flagged = False
        model_conf = None
        model_winner = None
        try:
            out = rt.compare(prompt=req.prompt, image_a_ref=req.imageA, image_b_ref=req.imageB, return_debug=False)
            model_conf = float(out.get("confidence", 0.0))
            model_winner = str(out.get("winner", "tie"))
            if model_conf > 0.9 and model_winner in ("A", "B") and model_winner != req.correct_winner:
                flagged = True
        except Exception:
            flagged = False

        payload = {
            "timestamp": time.time(),
            "prompt": req.prompt,
            "imageA": req.imageA,
            "imageB": req.imageB,
            "correct_winner": req.correct_winner,
            "model_winner": model_winner,
            "model_confidence": model_conf,
            "flagged": bool(flagged),
        }
        if flagged:
            _append_jsonl(os.path.join("logs", "flagged_feedback.jsonl"), payload)
        _append_jsonl(os.path.join("logs", "feedback.jsonl"), payload)
        return {"ok": True, "flagged": bool(flagged)}

    return app


app = create_app()
