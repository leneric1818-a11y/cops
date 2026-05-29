#!/usr/bin/env python3
"""Run, submit, and aggregate config-driven persona-control benchmarks.

This script provides a single benchmark interface for steering-style methods.
It supports:

1. planning a benchmark matrix
2. running the matrix locally
3. submitting each run as a parallel SLURM job
4. aggregating finished runs into a leaderboard

The benchmark config is JSON with this shape:

{
  "name": "steering_methods_v1",
  "direction_classifier": {
    "train_path": "outputs/metrics/steering_contrast_pairs_gpt54mini_200_flat.jsonl",
    "positive_label": "open",
    "negative_label": "defensive",
    "axis_name": "openness"
  },
  "runs": [
    {
      "name": "paired_dense_l11_open",
      "family": "paired_dense",
      "axis_name": "openness",
      "target_style": "open",
      "params": {
        "pair_data_path": "outputs/metrics/steering_contrast_pairs_gpt54mini_200.jsonl",
        "model_path": "Qwen/Qwen3-4B",
        "layers": "11",
        "alphas": "1.5,2.0"
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import importlib.util
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_PROJECT_DIR = "${CLUSTER_HOME}/llm_ft_comparison"
DEFAULT_CLUSTER_ENTRY_SCRIPT = "slurm/generated/steering_benchmark_entry_kiz0.sh"
DEFAULT_CLUSTER_PROJECTION_ENTRY_SCRIPT = "slurm/generated/projection_monitor_kiz0.sh"
DEFAULT_DIRECTION_CLASSIFIER = {
    "positive_label": "open",
    "negative_label": "defensive",
    "style_field": "style",
    "response_field": "response",
    "group_field": "seed_id",
}

DEFAULT_LLM_JUDGE = {
    "provider": "openai",
    "model": "gpt-5.4-mini",
    "temperature": 0.0,
    "max_examples": 100,
    "seed": 42,
    "max_retries": 3,
    "retry_sleep_seconds": 2.0,
    "rubric_version": "v1",
}

FAMILY_SCRIPT = {
    "prompt_diff": ROOT / "scripts" / "steering_vector_experiment.py",
    "paired_dense": ROOT / "scripts" / "paired_dense_steering_experiment.py",
    "reft_loreft": ROOT / "scripts" / "reft_loreft_experiment.py",
}

FAMILY_DEFAULTS = {
    "prompt_diff": {
        "trim_to_first_utterance": True,
        "model_path": "Qwen/Qwen3-4B",
        "prompt_format": "chat",
    },
    "paired_dense": {
        "trim_to_first_utterance": True,
        "model_path": "Qwen/Qwen3-4B",
        "prompt_format": "chat",
    },
    "reft_loreft": {
        "trim_to_first_utterance": True,
        "model_path": "Qwen/Qwen3-4B",
        "alpha": 1.0,
        "prompt_format": "chat",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or aggregate steering benchmarks.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "plan",
            "run-local",
            "submit-slurm",
            "submit-projection-slurm",
            "aggregate",
            "run-spec",
            "score-projection-spec",
        ),
        help="Benchmark action to perform.",
    )
    parser.add_argument(
        "--config",
        help="Benchmark JSON config for plan/run/submit/aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Benchmark output root. Defaults to outputs/metrics/benchmarks/<name>.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Explicit benchmark directory. Overrides --output-dir resolution.",
    )
    parser.add_argument(
        "--run-spec",
        default=None,
        help="Internal mode: path to one resolved run spec JSON.",
    )
    parser.add_argument(
        "--projection-spec",
        default=None,
        help="Internal mode: path to one resolved projection spec JSON.",
    )
    parser.add_argument(
        "--cluster-project-dir",
        default=DEFAULT_CLUSTER_PROJECT_DIR,
        help="Cluster checkout used by the SLURM entry script.",
    )
    parser.add_argument(
        "--cluster-entry-script",
        default=DEFAULT_CLUSTER_ENTRY_SCRIPT,
        help="Cluster entry script relative to the project root or absolute path.",
    )
    parser.add_argument(
        "--cluster-projection-entry-script",
        default=DEFAULT_CLUSTER_PROJECTION_ENTRY_SCRIPT,
        help="Cluster entry script used for projection-monitor SLURM jobs.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional sbatch partition override for submit-slurm.",
    )
    parser.add_argument(
        "--qos",
        default=None,
        help="Optional sbatch QoS override for submit-slurm.",
    )
    parser.add_argument(
        "--gres",
        default=None,
        help="Optional sbatch GRES override, for example 'gpu:0' or 'gpu:1'.",
    )
    parser.add_argument(
        "--run-names",
        nargs="*",
        default=None,
        help="Optional subset of run names to execute or submit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--inline-projection",
        action="store_true",
        help="Run projection monitoring inline during aggregate. By default aggregation only uses existing projection outputs.",
    )
    parser.add_argument(
        "--inline-judge",
        action="store_true",
        help="Run LLM-as-a-judge scoring inline during aggregate. By default aggregation only uses existing judge outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "run-spec":
        if not args.run_spec:
            raise SystemExit("--run-spec is required for --mode run-spec")
        spec = json.loads(Path(args.run_spec).read_text(encoding="utf-8"))
        execute_run_spec(spec, dry_run=args.dry_run)
        return

    if args.mode == "score-projection-spec":
        if not args.projection_spec:
            raise SystemExit("--projection-spec is required for --mode score-projection-spec")
        spec = json.loads(Path(args.projection_spec).read_text(encoding="utf-8"))
        execute_projection_spec(spec, dry_run=args.dry_run)
        return

    if not args.config:
        raise SystemExit("--config is required for this mode.")

    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    benchmark_dir = resolve_benchmark_dir(config, args)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = benchmark_dir / "manifest.json"

    selected_names = set(args.run_names) if args.run_names else None
    resolved_runs = resolve_runs(config, benchmark_dir, selected_names)
    manifest = build_manifest(config_path, benchmark_dir, config, resolved_runs)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "plan":
        print_plan(manifest)
        return

    if args.mode == "run-local":
        for run in manifest["runs"]:
            execute_run_spec(run, dry_run=args.dry_run)
        return

    if args.mode == "submit-slurm":
        submit_runs(manifest_path, args, dry_run=args.dry_run)
        return

    if args.mode == "submit-projection-slurm":
        submit_projection_run(manifest_path, args, dry_run=args.dry_run)
        return

    if args.mode == "aggregate":
        aggregate_benchmark(
            manifest_path,
            dry_run=args.dry_run,
            inline_projection=args.inline_projection,
            inline_judge=args.inline_judge,
        )
        return

    raise SystemExit(f"Unsupported mode: {args.mode}")


def resolve_benchmark_dir(config: dict, args: argparse.Namespace) -> Path:
    if args.benchmark_dir:
        return resolve_path(args.benchmark_dir)
    if args.output_dir:
        return resolve_path(args.output_dir)
    name = config.get("name")
    if not name:
        raise SystemExit("Benchmark config must include a top-level 'name'.")
    return ROOT / "outputs" / "metrics" / "benchmarks" / name


def resolve_runs(config: dict, benchmark_dir: Path, selected_names: set[str] | None) -> list[dict]:
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Benchmark config must include a non-empty 'runs' list.")
    shared_eval = config.get("shared_eval", {})
    if shared_eval and not isinstance(shared_eval, dict):
        raise SystemExit("'shared_eval' must be an object when provided.")
    shared_eval_manifest = shared_eval.get("manifest_path")
    shared_eval_limit = shared_eval.get("eval_limit")
    family_eval_params = shared_eval.get("family_params", {})

    resolved = []
    for run in runs:
        if not isinstance(run, dict):
            raise SystemExit("Each run config must be an object.")
        name = run.get("name")
        family = run.get("family")
        target_style = run.get("target_style")
        axis_name = run.get("axis_name") or config.get("axis_name")
        params = deepcopy(run.get("params", {}))
        if not name or not family or not target_style:
            raise SystemExit("Each run must define name, family, and target_style.")
        if family not in FAMILY_SCRIPT:
            raise SystemExit(f"Unsupported run family: {family}")
        if selected_names and name not in selected_names:
            continue

        resolved_params = deepcopy(FAMILY_DEFAULTS[family])
        if family_eval_params:
            resolved_params.update(deepcopy(family_eval_params.get(family, {})))
        if shared_eval_manifest is not None:
            resolved_params.setdefault("eval_manifest_path", shared_eval_manifest)
        if shared_eval_limit is not None:
            resolved_params.setdefault("eval_limit", shared_eval_limit)
        resolved_params.update(params)

        output_path = run.get("output_path")
        if output_path:
            output_path = str(resolve_path(output_path))
        else:
            output_path = str((benchmark_dir / "runs" / f"{name}.jsonl").resolve())

        save_vector_dir = run.get("save_vector_dir")
        if save_vector_dir:
            save_vector_dir = str(resolve_path(save_vector_dir))
        else:
            save_vector_dir = str((benchmark_dir / "vectors" / name).resolve())

        resolved_params["output_path"] = output_path
        resolved_params["save_vector_dir"] = save_vector_dir

        resolved.append(
            {
                "name": name,
                "family": family,
                "axis_name": axis_name,
                "target_style": target_style,
                "notes": run.get("notes"),
                "params": resolved_params,
                "output_path": output_path,
                "save_vector_dir": save_vector_dir,
            }
        )
    if not resolved:
        raise SystemExit("No runs selected after applying filters.")
    return resolved


def build_manifest(config_path: Path, benchmark_dir: Path, config: dict, resolved_runs: list[dict]) -> dict:
    ranking = deepcopy(config.get("ranking", {}))
    direction_classifier = resolve_direction_classifier(config)
    projection_monitor = resolve_projection_monitor(config)
    llm_judge = resolve_llm_judge(config)
    return {
        "benchmark_name": config["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "benchmark_dir": str(benchmark_dir.resolve()),
        "direction_classifier": direction_classifier,
        "projection_monitor": projection_monitor,
        "llm_judge": llm_judge,
        "classifier_train_path": direction_classifier["train_path"],
        "ranking": {
            "directional_weight": float(ranking.get("directional_weight", 1.0)),
            "content_weight": float(ranking.get("content_weight", 0.25)),
            "drift_weight": float(ranking.get("drift_weight", 0.25)),
            "significance_weight": float(ranking.get("significance_weight", 0.0)),
            "significance_threshold": float(ranking.get("significance_threshold", 0.05)),
        },
        "runs": resolved_runs,
    }


def print_plan(manifest: dict) -> None:
    classifier = manifest["direction_classifier"]
    print(f"Benchmark: {manifest['benchmark_name']}")
    print(f"Benchmark dir: {manifest['benchmark_dir']}")
    print(f"Direction classifier train path: {classifier['train_path']}")
    print(
        f"Direction axis: {classifier['axis_name']} "
        f"({classifier['negative_label']} -> {classifier['positive_label']})"
    )
    if manifest.get("projection_monitor"):
        print(f"Projection monitor vector: {manifest['projection_monitor']['vector_path']}")
    if manifest.get("llm_judge"):
        print(
            f"LLM judge: {manifest['llm_judge']['provider']} / "
            f"{manifest['llm_judge']['model']} ({manifest['llm_judge']['rubric_version']})"
        )
    print("")
    for run in manifest["runs"]:
        axis_suffix = f" [{run['axis_name']}]" if run.get("axis_name") else ""
        print(f"- {run['name']} [{run['family']}] -> {run['target_style']}{axis_suffix}")
        print(f"  output: {run['output_path']}")
        print(f"  params: {json.dumps(run['params'], ensure_ascii=False, sort_keys=True)}")


def execute_run_spec(spec: dict, dry_run: bool = False) -> None:
    family = spec["family"]
    script_path = FAMILY_SCRIPT[family]
    params = deepcopy(spec["params"])
    output_path = Path(params["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector_dir = Path(params["save_vector_dir"])
    vector_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-B", str(script_path)]
    cmd.extend(build_cli_args(params))

    print(f"\n[{spec['name']}]")
    print(" ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return

    subprocess.run(cmd, cwd=str(ROOT), check=True)


def execute_projection_spec(spec: dict, dry_run: bool = False) -> None:
    cmd = build_projection_command_from_spec(spec)
    if cmd is None:
        raise SystemExit("Projection spec is missing required projection-monitor settings.")

    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[projection::{spec['benchmark_name']}]")
    print(" ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return

    subprocess.run(cmd, cwd=str(ROOT), check=True)


def build_cli_args(params: dict) -> list[str]:
    args = []
    for key, value in params.items():
        option = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            args.append(option if value else f"--no-{key.replace('_', '-')}")
            continue
        if isinstance(value, dict):
            encoded = ",".join(f"{sub_key}:{sub_value}" for sub_key, sub_value in value.items())
            args.extend([option, encoded])
            continue
        if isinstance(value, (list, tuple)):
            encoded = ",".join(str(item) for item in value)
            args.extend([option, encoded])
            continue
        args.extend([option, str(value)])
    return args


def submit_runs(manifest_path: Path, args: argparse.Namespace, dry_run: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_dir = Path(manifest["benchmark_dir"])
    job_specs_dir = benchmark_dir / "job_specs"
    job_specs_dir.mkdir(parents=True, exist_ok=True)

    cluster_project_dir = Path(os.path.expandvars(os.path.expanduser(args.cluster_project_dir)))
    entry_script = Path(args.cluster_entry_script)
    if not entry_script.is_absolute():
        entry_script = cluster_project_dir / entry_script

    updated_runs = []
    for run in manifest["runs"]:
        run_spec_path = job_specs_dir / f"{run['name']}.json"
        cluster_run = deepcopy(run)
        cluster_run["project_dir"] = str(cluster_project_dir)
        cluster_run["output_path"] = rewrite_local_path_for_cluster(
            run["output_path"],
            cluster_project_dir,
            benchmark_dir,
        )
        cluster_run["save_vector_dir"] = rewrite_local_path_for_cluster(
            run["save_vector_dir"],
            cluster_project_dir,
            benchmark_dir,
        )
        cluster_run["params"]["output_path"] = cluster_run["output_path"]
        cluster_run["params"]["save_vector_dir"] = cluster_run["save_vector_dir"]
        run_spec_path.write_text(json.dumps(cluster_run, ensure_ascii=False, indent=2), encoding="utf-8")

        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "--job-name",
            sanitize_job_name(run["name"]),
            "--export",
            f"ALL,RUN_SPEC_PATH={run_spec_path}",
        ]
        if args.partition:
            sbatch_cmd.extend(["--partition", args.partition])
        if args.qos:
            sbatch_cmd.extend(["--qos", args.qos])
        if args.gres:
            sbatch_cmd.extend(["--gres", args.gres])
        sbatch_cmd.append(str(entry_script))

        print("\nSubmitting:", " ".join(shlex.quote(part) for part in sbatch_cmd))
        job_id = None
        if not dry_run:
            result = subprocess.run(
                sbatch_cmd,
                cwd=str(cluster_project_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            print(f"Submitted {run['name']} as job {job_id}")

        updated = deepcopy(run)
        updated["run_spec_path"] = str(run_spec_path)
        updated["job_id"] = job_id
        updated_runs.append(updated)

    manifest["submitted_at"] = datetime.now(timezone.utc).isoformat()
    manifest["runs"] = updated_runs
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def submit_projection_run(manifest_path: Path, args: argparse.Namespace, dry_run: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("projection_monitor"):
        raise SystemExit("Benchmark manifest has no projection_monitor section.")

    benchmark_dir = Path(manifest["benchmark_dir"])
    cluster_project_dir = Path(os.path.expandvars(os.path.expanduser(args.cluster_project_dir)))
    cluster_benchmark_dir = Path(
        rewrite_local_path_for_cluster(str(benchmark_dir), cluster_project_dir, benchmark_dir)
    )
    projection_specs_dir = cluster_benchmark_dir / "projection_job_specs"
    entry_script = Path(args.cluster_projection_entry_script)
    if not entry_script.is_absolute():
        entry_script = cluster_project_dir / entry_script

    projection_spec_path = projection_specs_dir / "projection.json"
    cluster_spec = build_cluster_projection_spec(manifest, cluster_project_dir)
    if not dry_run:
        projection_specs_dir.mkdir(parents=True, exist_ok=True)
        projection_spec_path.write_text(
            json.dumps(cluster_spec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        sanitize_job_name(f"proj_{manifest['benchmark_name']}"),
        "--export",
        f"ALL,PROJECTION_SPEC_PATH={projection_spec_path}",
    ]
    if args.partition:
        sbatch_cmd.extend(["--partition", args.partition])
    if args.qos:
        sbatch_cmd.extend(["--qos", args.qos])
    if args.gres:
        sbatch_cmd.extend(["--gres", args.gres])
    sbatch_cmd.append(str(entry_script))

    print("\nSubmitting projection monitor:", " ".join(shlex.quote(part) for part in sbatch_cmd))
    job_id = None
    if not dry_run:
        result = subprocess.run(
            sbatch_cmd,
            cwd=str(cluster_project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        print(f"Submitted projection monitor for {manifest['benchmark_name']} as job {job_id}")

    manifest["projection_submitted_at"] = datetime.now(timezone.utc).isoformat()
    manifest["projection_job_id"] = job_id
    manifest["projection_spec_path"] = str(projection_spec_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_benchmark(
    manifest_path: Path,
    dry_run: bool = False,
    inline_projection: bool = False,
    inline_judge: bool = False,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_dir = Path(manifest["benchmark_dir"])
    score_paths = [Path(run["output_path"]).resolve() for run in manifest["runs"] if Path(run["output_path"]).exists()]
    missing_paths = [run["output_path"] for run in manifest["runs"] if not Path(run["output_path"]).exists()]

    if not score_paths:
        raise SystemExit("No completed run outputs found to aggregate.")

    if missing_paths:
        print("Skipping missing outputs:")
        for path in missing_paths:
            print(f"- {path}")

    classifier_dir = benchmark_dir / "scoring" / "classifier"
    content_dir = benchmark_dir / "scoring" / "content"
    projection_dir = benchmark_dir / "scoring" / "projection"
    judge_dir = benchmark_dir / "scoring" / "judge"
    classifier = manifest["direction_classifier"]
    classifier_cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "score_style_axis_classifier.py"),
        "--train-path",
        classifier["train_path"],
        "--output-dir",
        str(classifier_dir),
        "--positive-label",
        classifier["positive_label"],
        "--negative-label",
        classifier["negative_label"],
        "--style-field",
        classifier["style_field"],
        "--response-field",
        classifier["response_field"],
        "--group-field",
        classifier["group_field"],
        "--axis-name",
        classifier["axis_name"],
        "--score-paths",
        *[str(path) for path in score_paths],
    ]
    content_cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "score_content_preservation.py"),
        "--output-dir",
        str(content_dir),
        "--score-paths",
        *[str(path) for path in score_paths],
    ]

    print("\nClassifier scoring:")
    print(" ".join(shlex.quote(part) for part in classifier_cmd))
    print("\nContent scoring:")
    print(" ".join(shlex.quote(part) for part in content_cmd))
    projection_cmd = build_projection_command(manifest, projection_dir, score_paths)
    if projection_cmd and inline_projection:
        print("\nProjection scoring:")
        print(" ".join(shlex.quote(part) for part in projection_cmd))
    elif projection_cmd:
        print("\nProjection scoring:")
        print("Skipping inline projection during aggregate.")
        print("Use --mode submit-projection-slurm to score projection on the cluster.")
    judge_cmd = build_judge_command(manifest, manifest_path, judge_dir, score_paths)
    if judge_cmd and inline_judge:
        print("\nLLM judge scoring:")
        print(" ".join(shlex.quote(part) for part in judge_cmd))
    elif judge_cmd:
        print("\nLLM judge scoring:")
        print("Skipping inline LLM judge scoring during aggregate.")
        print("Re-run aggregate with --inline-judge once you want to spend judge tokens.")
    if not dry_run:
        subprocess.run(classifier_cmd, cwd=str(ROOT), check=True)
        subprocess.run(content_cmd, cwd=str(ROOT), check=True)
        projection_available = (projection_dir / "all_runs_summary.csv").exists()
        judge_available = (judge_dir / "all_runs_summary.csv").exists()
        if projection_cmd and inline_projection:
            try:
                subprocess.run(projection_cmd, cwd=str(ROOT), check=True)
                projection_available = True
            except subprocess.CalledProcessError as exc:
                print(
                    "\nWARNING: projection scoring failed; continuing with classifier/content-only leaderboard."
                )
                print(exc)
        if judge_cmd and inline_judge:
            try:
                subprocess.run(judge_cmd, cwd=str(ROOT), check=True)
                judge_available = True
            except subprocess.CalledProcessError as exc:
                print(
                    "\nWARNING: LLM judge scoring failed; continuing with existing leaderboard metrics."
                )
                print(exc)
        build_leaderboard(
            manifest,
            classifier_dir,
            content_dir,
            projection_dir if projection_available else None,
            judge_dir if judge_available else None,
        )


def build_leaderboard(
    manifest: dict,
    classifier_dir: Path,
    content_dir: Path,
    projection_dir: Path | None = None,
    judge_dir: Path | None = None,
) -> None:
    ranking = manifest.get("ranking", {})
    directional_weight = float(ranking.get("directional_weight", 1.0))
    content_weight = float(ranking.get("content_weight", 0.25))
    drift_weight = float(ranking.get("drift_weight", 0.25))
    significance_weight = float(ranking.get("significance_weight", 0.0))
    significance_threshold = float(ranking.get("significance_threshold", 0.05))

    classifier_rows = load_csv_rows(classifier_dir / "all_runs_summary.csv")
    content_rows = load_csv_rows(content_dir / "all_runs_summary.csv")
    projection_rows = (
        load_csv_rows(projection_dir / "all_runs_summary.csv")
        if projection_dir and (projection_dir / "all_runs_summary.csv").exists()
        else []
    )
    judge_rows = (
        load_csv_rows(judge_dir / "all_runs_summary.csv")
        if judge_dir and (judge_dir / "all_runs_summary.csv").exists()
        else []
    )
    content_by_key = {
        (row["source_file"], row["config_id"]): row
        for row in content_rows
    }
    projection_by_key = {
        (row["source_file"], row["config_id"]): row
        for row in projection_rows
    }
    judge_by_key = {
        (row["source_file"], row["config_id"]): row
        for row in judge_rows
    }
    run_by_source = {
        str(Path(run["output_path"]).resolve()): run
        for run in manifest["runs"]
    }
    classifier = manifest["direction_classifier"]
    positive_label = classifier["positive_label"]
    negative_label = classifier["negative_label"]

    leaderboard = []
    for row in classifier_rows:
        source_file = row["source_file"]
        run = run_by_source.get(source_file)
        if run is None:
            continue
        content = content_by_key.get((source_file, row["config_id"]), {})
        projection = projection_by_key.get((source_file, row["config_id"]), {})
        judge = judge_by_key.get((source_file, row["config_id"]), {})
        target_style = run["target_style"]
        mean_delta = float(row["mean_delta_vs_base"])
        if target_style == positive_label:
            directional_effect = mean_delta
        elif target_style == negative_label:
            directional_effect = -mean_delta
        else:
            raise SystemExit(
                f"Run {run['name']} target_style={target_style} does not match "
                f"classifier labels {negative_label}/{positive_label}."
            )
        wilcoxon_p = float(row["wilcoxon_p"])
        content_score = float(content.get("content_preservation_score", 0.0))
        drift_rate = float(content.get("drift_flag_rate", 0.0))
        significance_bonus = (
            significance_weight
            if directional_effect > 0 and wilcoxon_p <= significance_threshold
            else 0.0
        )
        rank_score = (
            directional_weight * directional_effect
            + content_weight * content_score
            - drift_weight * drift_rate
            + significance_bonus
        )
        leaderboard.append(
            {
                "run_name": run["name"],
                "family": run["family"],
                "axis_name": run.get("axis_name") or classifier["axis_name"],
                "target_style": target_style,
                "config_id": row["config_id"],
                "source_file": source_file,
                "directional_effect": directional_effect,
                "mean_delta_vs_base": mean_delta,
                "wilcoxon_p": wilcoxon_p,
                "wins": int(float(row["wins"])),
                "losses": int(float(row["losses"])),
                "ties": int(float(row["ties"])),
                "content_preservation_score": content_score,
                "drift_flag_rate": drift_rate,
                "mean_token_jaccard_vs_base": float(content.get("mean_token_jaccard_vs_base", 0.0)),
                "mean_novelty_ratio": float(content.get("mean_novelty_ratio", 0.0)),
                "numeric_change_rate": float(content.get("numeric_change_rate", 0.0)),
                "yesno_flip_rate": float(content.get("yesno_flip_rate", 0.0)),
                "projection_mean_delta_vs_base": float(projection.get("mean_delta_vs_base", 0.0)),
                "projection_wilcoxon_p": float(projection.get("wilcoxon_p", 1.0)),
                "judge_n_examples": int(float(judge.get("n_examples", 0.0))),
                "judge_pairwise_steered_win_rate": float(
                    judge.get("pairwise_steered_win_rate", 0.0)
                ),
                "judge_pairwise_base_win_rate": float(judge.get("pairwise_base_win_rate", 0.0)),
                "judge_pairwise_tie_rate": float(judge.get("pairwise_tie_rate", 0.0)),
                "judge_mean_axis_alignment_base": float(
                    judge.get("mean_axis_alignment_base", 0.0)
                ),
                "judge_mean_axis_alignment_steered": float(
                    judge.get("mean_axis_alignment_steered", 0.0)
                ),
                "judge_mean_axis_alignment_delta": float(
                    judge.get("mean_axis_alignment_delta", 0.0)
                ),
                "judge_mean_case_fidelity_base": float(
                    judge.get("mean_case_fidelity_base", 0.0)
                ),
                "judge_mean_case_fidelity_steered": float(
                    judge.get("mean_case_fidelity_steered", 0.0)
                ),
                "judge_mean_case_fidelity_delta": float(
                    judge.get("mean_case_fidelity_delta", 0.0)
                ),
                "judge_mean_client_role_fidelity_base": float(
                    judge.get("mean_client_role_fidelity_base", 0.0)
                ),
                "judge_mean_client_role_fidelity_steered": float(
                    judge.get("mean_client_role_fidelity_steered", 0.0)
                ),
                "judge_mean_client_role_fidelity_delta": float(
                    judge.get("mean_client_role_fidelity_delta", 0.0)
                ),
                "judge_mean_training_utility_base": float(
                    judge.get("mean_training_utility_base", 0.0)
                ),
                "judge_mean_training_utility_steered": float(
                    judge.get("mean_training_utility_steered", 0.0)
                ),
                "judge_mean_training_utility_delta": float(
                    judge.get("mean_training_utility_delta", 0.0)
                ),
                "significance_bonus": significance_bonus,
                "rank_score": rank_score,
            }
        )

    leaderboard.sort(key=lambda item: (item["rank_score"], item["directional_effect"]), reverse=True)
    leaderboard_path = Path(manifest["benchmark_dir"]) / "leaderboard.csv"
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)

    (Path(manifest["benchmark_dir"]) / "leaderboard.json").write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTop leaderboard rows:")
    for row in leaderboard[:10]:
        print(
            row["run_name"],
            row["config_id"],
            "direction",
            round(row["directional_effect"], 4),
            "content",
            round(row["content_preservation_score"], 4),
            "drift",
            round(row["drift_flag_rate"], 4),
            "rank",
            round(row["rank_score"], 4),
        )


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_direction_classifier(config: dict) -> dict:
    spec = deepcopy(config.get("direction_classifier") or {})
    legacy_train_path = config.get("classifier_train_path")
    if not spec and not legacy_train_path:
        raise SystemExit(
            "Benchmark config must define either 'direction_classifier' or legacy 'classifier_train_path'."
        )
    if legacy_train_path and "train_path" not in spec:
        spec["train_path"] = legacy_train_path
    resolved = deepcopy(DEFAULT_DIRECTION_CLASSIFIER)
    resolved.update(spec)
    if "train_path" not in resolved:
        raise SystemExit("Direction classifier config must define 'train_path'.")
    resolved["train_path"] = str(resolve_path(resolved["train_path"]))
    resolved["axis_name"] = resolved.get("axis_name") or f"{resolved['negative_label']}_vs_{resolved['positive_label']}"
    return resolved


def resolve_projection_monitor(config: dict) -> dict | None:
    spec = deepcopy(config.get("projection_monitor") or {})
    if not spec:
        return None
    if "vector_path" not in spec:
        raise SystemExit("Projection monitor config must define 'vector_path'.")
    spec["vector_path"] = str(resolve_path(spec["vector_path"]))
    if spec.get("model_path"):
        spec["model_path"] = str(spec["model_path"])
    if spec.get("adapter_path"):
        spec["adapter_path"] = str(resolve_path(spec["adapter_path"]))
    if spec.get("batch_size") is not None:
        spec["batch_size"] = int(spec["batch_size"])
    return spec


def resolve_llm_judge(config: dict) -> dict | None:
    spec = deepcopy(config.get("llm_judge") or {})
    if not spec:
        return None
    resolved = deepcopy(DEFAULT_LLM_JUDGE)
    resolved.update(spec)
    provider = str(resolved.get("provider", "openai")).strip().lower()
    if provider != "openai":
        raise SystemExit(f"Unsupported llm_judge provider: {provider}")
    if not resolved.get("model"):
        raise SystemExit("llm_judge config must define 'model'.")
    resolved["provider"] = provider
    resolved["model"] = str(resolved["model"])
    resolved["temperature"] = float(resolved.get("temperature", 0.0))
    resolved["max_examples"] = int(resolved["max_examples"]) if resolved.get("max_examples") else None
    resolved["seed"] = int(resolved.get("seed", 42))
    resolved["max_retries"] = int(resolved.get("max_retries", 3))
    resolved["retry_sleep_seconds"] = float(resolved.get("retry_sleep_seconds", 2.0))
    resolved["rubric_version"] = str(resolved.get("rubric_version", "v1"))
    return resolved


def build_projection_command(manifest: dict, projection_dir: Path, score_paths: list[Path]) -> list[str] | None:
    projection = manifest.get("projection_monitor")
    if not projection:
        return None
    if importlib.util.find_spec("accelerate") is None:
        print("Skipping projection scoring locally because 'accelerate' is not installed.")
        return None
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "score_persona_monitoring.py"),
        "--vector-path",
        projection["vector_path"],
        "--output-dir",
        str(projection_dir),
        "--score-paths",
        *[str(path) for path in score_paths],
    ]
    if projection.get("model_path"):
        cmd.extend(["--model-path", projection["model_path"]])
    if projection.get("adapter_path"):
        cmd.extend(["--adapter-path", projection["adapter_path"]])
    if projection.get("batch_size"):
        cmd.extend(["--batch-size", str(projection["batch_size"])])
    return cmd


def build_judge_command(
    manifest: dict,
    manifest_path: Path,
    judge_dir: Path,
    score_paths: list[Path],
) -> list[str] | None:
    judge = manifest.get("llm_judge")
    if not judge:
        return None
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "score_llm_judge.py"),
        "--benchmark-manifest",
        str(manifest_path),
        "--output-dir",
        str(judge_dir),
        "--provider",
        judge["provider"],
        "--model",
        judge["model"],
        "--temperature",
        str(judge["temperature"]),
        "--seed",
        str(judge["seed"]),
        "--max-retries",
        str(judge["max_retries"]),
        "--retry-sleep-seconds",
        str(judge["retry_sleep_seconds"]),
        "--rubric-version",
        judge["rubric_version"],
        "--score-paths",
        *[str(path) for path in score_paths],
    ]
    if judge.get("max_examples"):
        cmd.extend(["--max-examples", str(judge["max_examples"])])
    return cmd


def build_projection_command_from_spec(spec: dict) -> list[str] | None:
    projection = spec.get("projection_monitor")
    if not projection:
        return None
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "score_persona_monitoring.py"),
        "--vector-path",
        projection["vector_path"],
        "--output-dir",
        spec["output_dir"],
        "--score-paths",
        *spec["score_paths"],
    ]
    if projection.get("model_path"):
        cmd.extend(["--model-path", projection["model_path"]])
    if projection.get("adapter_path"):
        cmd.extend(["--adapter-path", projection["adapter_path"]])
    if projection.get("batch_size"):
        cmd.extend(["--batch-size", str(projection["batch_size"])])
    return cmd


def build_cluster_projection_spec(manifest: dict, cluster_project_dir: Path) -> dict:
    benchmark_dir = Path(manifest["benchmark_dir"])
    score_paths = [run["output_path"] for run in manifest["runs"]]
    projection = deepcopy(manifest["projection_monitor"])
    projection["vector_path"] = rewrite_local_path_for_cluster(
        projection["vector_path"],
        cluster_project_dir,
        benchmark_dir,
    )
    if projection.get("adapter_path"):
        projection["adapter_path"] = rewrite_local_path_for_cluster(
            projection["adapter_path"],
            cluster_project_dir,
            benchmark_dir,
        )
    output_dir = rewrite_local_path_for_cluster(
        str(benchmark_dir / "scoring" / "projection"),
        cluster_project_dir,
        benchmark_dir,
    )
    return {
        "benchmark_name": manifest["benchmark_name"],
        "benchmark_dir": rewrite_local_path_for_cluster(str(benchmark_dir), cluster_project_dir, benchmark_dir),
        "output_dir": output_dir,
        "score_paths": [
            rewrite_local_path_for_cluster(path, cluster_project_dir, benchmark_dir)
            for path in score_paths
        ],
        "projection_monitor": projection,
    }


def rewrite_local_path_for_cluster(path: str, cluster_project_dir: Path, local_benchmark_dir: Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
        return str((cluster_project_dir / relative).resolve())
    except ValueError:
        try:
            relative = resolved.relative_to(local_benchmark_dir)
            cluster_benchmark_dir = cluster_project_dir / "outputs" / "metrics" / "benchmarks" / local_benchmark_dir.name
            return str((cluster_benchmark_dir / relative).resolve())
        except ValueError:
            return str(resolved)


def sanitize_job_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    return cleaned[:80] or "steering_benchmark"


def resolve_path(path_value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    main()
