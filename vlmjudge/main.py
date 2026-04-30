"""
author: Jayesh Pandey
summary: Modular entrypoint for vlmjudge, providing CLI for scoring and comparing images with multi-scorer support and VLM fusion.
"""

from __future__ import annotations

import argparse
import sys
import logging
import json
import time
from pathlib import Path
from typing import Optional

from vlmjudge.scorers.image_reward import ImageRewardScorer, ImageRewardScorerConfig
from vlmjudge.scorers.clip_score import OpenCLIPScorer
from vlmjudge.scorers.aesthetic import AestheticScorer
from vlmjudge.scorers.lpips import LPIPSScorer
from vlmjudge.pipelines.compare_pipeline import ComparePipeline, ComparePipelineConfig
from vlmjudge.datasets.quality import filter_samples, split_dataset, quality_report
from vlmjudge.vlm.ensemble import VLMEnsemble


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score or compare images using multiple scorers (Phase 3).")
    parser.add_argument("--image", default=None, type=str, help="Path to an input image file.")
    parser.add_argument("--image-b", default=None, type=str, help="Optional second image path (enables compare mode).")
    parser.add_argument("--prompt", default=None, type=str, help="Text prompt to score/compare against.")
    parser.add_argument("--batch", default=None, type=str, help="Path to a JSON list of {prompt,imgA,imgB} for batch compare.")
    parser.add_argument("--out", default=None, type=str, help="Where to write full comparison results JSON (batch mode).")
    parser.add_argument("--dataset-out", default=None, type=str, help="Where to write preference dataset JSON (batch mode).")
    parser.add_argument("--quality-filter", default="medium", type=str, help='Minimum quality to keep: "low"|"medium"|"high" (batch mode).')
    parser.add_argument("--quality-report", action="store_true", help="Print quality stats (batch mode).")
    parser.add_argument("--min-coverage", default=0.5, type=float, help="Minimum scorer coverage to keep (batch mode).")
    parser.add_argument("--threshold", default=0.05, type=float, help="Tie threshold on aggregated score delta.")
    parser.add_argument("--device", default=None, type=str, help='Device override, e.g. "cpu" or "cuda:0".')
    parser.add_argument("--download-root", default=None, type=str, help="Cache directory for downloaded weights.")
    parser.add_argument("--med-config", default=None, type=str, help="Path to med_config.json override.")
    parser.add_argument("--model-name", default="ImageReward-v1.0", type=str, help="ImageReward model name or path.")
    parser.add_argument("--use-vlm", action="store_true", help="Enable Qwen2.5-VL judge for fusion in compare mode.")
    parser.add_argument("--vlm-runs", default=3, type=int, help="Number of VLM voting runs per pair (Phase 4.5).")
    parser.add_argument("--vlm-max-new-tokens", default=192, type=int, help="Max new tokens for VLM output (Phase 4.5).")
    parser.add_argument("--log-level", default="WARNING", type=str, help="Logging level (e.g., INFO, WARNING).")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING))

    scorers = {}

    # Construct scorers defensively: a scorer may be "disabled" internally if deps/weights are missing.
    scorers["image_reward"] = ImageRewardScorer(
        ImageRewardScorerConfig(
            model_name=args.model_name,
            device=args.device,
            download_root=args.download_root,
            med_config=args.med_config,
        )
    )
    scorers["openclip"] = OpenCLIPScorer()
    scorers["aesthetic"] = AestheticScorer()
    scorers["lpips"] = LPIPSScorer()

    log = logging.getLogger(__name__)

    vlm_ensemble = None
    if args.use_vlm:
        log.info("Initializing VLM Ensemble (Multi-Model Support)")
        vlm_ensemble = VLMEnsemble(
            config={
                "vlm_runs": int(args.vlm_runs),
                "vlm_max_new_tokens": int(args.vlm_max_new_tokens)
            },
            strict=False,
        )

    # Batch compare mode
    if args.batch is not None:
        t0 = time.perf_counter()
        batch_path = Path(args.batch)
        # PowerShell often writes UTF-8 with BOM; utf-8-sig handles both BOM and non-BOM.
        items = json.loads(batch_path.read_text(encoding="utf-8-sig"))
        if not isinstance(items, list):
            raise ValueError("--batch input must be a JSON list.")

        pipeline = ComparePipeline(
            scorers,
            config=ComparePipelineConfig(threshold=float(args.threshold), vlm_runs=int(args.vlm_runs)),
            vlm_judge=vlm_ensemble,
        )

        outputs = []
        dataset_entries = []
        for i, item in enumerate(items):
            prompt = item.get("prompt")
            imgA = item.get("imgA")
            imgB = item.get("imgB")
            if not prompt or not imgA or not imgB:
                log.warning("event=batch_skip idx=%d reason=missing_fields", i)
                continue
            outputs.append(pipeline.run(imgA, imgB, prompt))
            dataset_entries.append(outputs[-1]["dataset_entry"])

        # Write outputs (defaults created in batch mode to make dataset generation easy).
        out_path = Path(args.out) if args.out else batch_path.with_name(batch_path.stem + ".comparisons.json")
        dataset_out_path = (
            Path(args.dataset_out) if args.dataset_out else batch_path.with_name(batch_path.stem + ".preferences.json")
        )
        out_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")

        # Quality filtering + splitting
        report = quality_report(dataset_entries)
        if args.quality_report:
            print(report["summary"])
            print(
                f"Avg confidence: {report['avg_confidence']:.3f} | Avg delta: {report['avg_delta']:.3f} | Avg disagreement: {report['avg_disagreement']:.3f}"
            )

        filtered = filter_samples(
            dataset_entries,
            min_quality=str(args.quality_filter),
            min_coverage=float(args.min_coverage),
        )
        dataset_out_path.write_text(json.dumps(filtered, indent=2, sort_keys=True), encoding="utf-8")

        high, medium, low = split_dataset(dataset_entries)
        stem = dataset_out_path.with_suffix("").name
        base_dir = dataset_out_path.parent
        (base_dir / f"{stem}.high.json").write_text(json.dumps(high, indent=2, sort_keys=True), encoding="utf-8")
        (base_dir / f"{stem}.medium.json").write_text(json.dumps(medium, indent=2, sort_keys=True), encoding="utf-8")
        (base_dir / f"{stem}.low.json").write_text(json.dumps(low, indent=2, sort_keys=True), encoding="utf-8")

        dt_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "event=batch_done n=%d ms=%.2f out=%s dataset_out=%s",
            len(outputs),
            dt_ms,
            str(out_path),
            str(dataset_out_path),
        )
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0

    # Compare mode
    if args.image is not None and args.image_b is not None:
        if not args.prompt:
            raise ValueError("--prompt is required for compare mode.")
        pipeline = ComparePipeline(
            scorers,
            config=ComparePipelineConfig(threshold=float(args.threshold), vlm_runs=int(args.vlm_runs)),
            vlm_judge=vlm_ensemble,
        )
        result = pipeline.run(args.image, args.image_b, args.prompt)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    # Score-only mode (single image)
    if args.image is None:
        raise ValueError("Provide --image for scoring, or --image + --image-b for compare mode, or --batch for batch mode.")
    if not args.prompt:
        raise ValueError("--prompt is required for score-only mode.")

    results = {}
    for name, scorer in scorers.items():
        try:
            results[name] = scorer.score(args.image, prompt=args.prompt, image_b=None)
        except Exception as e:
            log.warning("event=scorer_crash scorer=%s err=%s", name, e)
            results[name] = {"score": 0.5, "confidence": 0.0}

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
