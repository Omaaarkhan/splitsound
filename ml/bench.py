"""Benchmark the exported graphs with onnxruntime on this machine.

    uv run --locked python bench.py                         # everything, writes dist/reports/bench.json
    uv run --locked python bench.py --models scnet_large --suites cpu coreml
    uv run --locked python bench.py --quick

Every configuration runs in its own child process, so that session creation
time, steady-state time per segment and peak resident memory are measured in
isolation, and so that a provider that crashes or hangs (CoreML compilation
of a large graph can take minutes) cannot take the run down with it.

Suites:
  cpu       CPU execution provider at several intra-op thread counts
  opt       graph optimisation levels (disabled / basic / extended / all), and the effect of
            leaving the batch dimension symbolic instead of fixing it to 1
  coreml    CoreML execution provider: MLProgram and NeuralNetwork formats, compute units ALL
            and CPUAndGPU, with the number of graph partitions / nodes CoreML actually took

Realtime factor = seconds of audio in one segment / seconds to run the graph on it.
The host-side STFT and inverse STFT are not included (they are a few percent;
see BENCHMARKS.md), nor is the chunk overlap: with stride S and segment N a
whole track costs N / S graph runs per segment of audio.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

MODELS = {
    # id: (path under dist/, seconds of audio per segment)
    "scnet_small": ("scnet_small.onnx", 11.0),
    "scnet_large": ("scnet_large.onnx", 11.0),
    "htdemucs_6s": ("htdemucs_6s.onnx", 7.8),
    "melband_roformer_vocals": ("melband_roformer_vocals.onnx", None),
    "efficientat_mn10_as": ("efficientat_mn10_as.onnx", 10.0),
}


def child(config: dict) -> None:
    """Run one configuration and print one JSON line."""
    import numpy as np
    import onnxruntime as ort

    import reference
    import synth

    path = ML_DIR / "dist" / config["file"]
    options = ort.SessionOptions()
    options.log_severity_level = 1 if config.get("verbose") else 3
    if config.get("threads"):
        options.intra_op_num_threads = config["threads"]
    level = config.get("optimization", "all")
    options.graph_optimization_level = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[level]
    if config.get("fixBatch", True):
        options.add_free_dimension_override_by_name("batch", 1)
    if config.get("memPattern") is False:
        options.enable_mem_pattern = False
    if config.get("cpuArena") is False:
        options.enable_cpu_mem_arena = False
    providers = ["CPUExecutionProvider"]
    if config.get("coreml"):
        providers = [("CoreMLExecutionProvider", config["coreml"]), "CPUExecutionProvider"]

    started = time.perf_counter()
    session = ort.InferenceSession(str(path), options, providers=providers)
    create_seconds = time.perf_counter() - started
    metadata = session.get_modelmeta().custom_metadata_map

    # Realistic inputs: the synthetic clip through the real host-side transform.
    kind = metadata["kind"]
    if kind == "tagger":
        feeds = {"waveform": synth.synth_clip(10.0, sample_rate=32000).mean(axis=0)[np.newaxis].astype(np.float32)}
    elif kind == "htdemucs":
        segment = synth.synth_clip(12.0)[:, : reference.HT_SEGMENT]
        feeds = {"mix": segment[np.newaxis], "spec": reference.htdemucs_stft(segment)[np.newaxis]}
    elif kind == "scnet":
        segment = synth.synth_clip(12.0)[:, : reference.SC_SEGMENT]
        feeds = {"spec": reference.scnet_stft(segment)[np.newaxis]}
    else:
        segment = synth.synth_clip(12.0)[:, : int(metadata["segmentSamples"])]
        feeds = {"spec": reference.melband_stft(segment)[np.newaxis]}

    started = time.perf_counter()
    first = session.run(None, feeds)
    first_seconds = time.perf_counter() - started
    times = []
    for _ in range(config.get("runs", 3)):
        started = time.perf_counter()
        session.run(None, feeds)
        times.append(time.perf_counter() - started)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024                                    # Linux reports kilobytes, macOS bytes
    print("RESULT " + json.dumps({
        "createSeconds": create_seconds, "firstRunSeconds": first_seconds,
        "runSeconds": times, "bestSeconds": min(times), "medianSeconds": sorted(times)[len(times) // 2],
        "peakRssBytes": peak, "providers": session.get_providers(),
        "outputChecksum": float(np.abs(first[0]).mean()),
    }), flush=True)


def run_config(config: dict, timeout: int) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(config)]
    started = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**config, "error": f"timeout after {timeout} s"}
    result = dict(config)
    result["wallSeconds"] = time.time() - started
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            result.update(json.loads(line[7:]))
    if "bestSeconds" not in result:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        result["error"] = " | ".join(line[:300] for line in tail) or f"exit status {completed.returncode}"
    # CoreML reports what it took from the graph at verbose log level.
    match = re.findall(r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) "
                       r"number of nodes supported by CoreML: (\d+)", completed.stderr + completed.stdout)
    if match:
        partitions, nodes, supported = map(int, match[-1])
        result["coremlPartitions"], result["graphNodes"], result["coremlNodes"] = partitions, nodes, supported
    return result


def describe(result: dict, audio_seconds: float | None) -> str:
    if "error" in result:
        return f"FAILED: {result['error'][:200]}"
    text = (f"create {result['createSeconds']:6.2f} s  first {result['firstRunSeconds']:6.2f} s  "
            f"best {result['bestSeconds']:6.3f} s  median {result['medianSeconds']:6.3f} s  "
            f"peak RSS {result['peakRssBytes'] / 2**20:6.0f} MiB")
    if audio_seconds:
        text += f"  realtime x{audio_seconds / result['bestSeconds']:5.1f}"
    if "coremlNodes" in result:
        text += f"  CoreML took {result['coremlNodes']}/{result['graphNodes']} nodes in {result['coremlPartitions']} partitions"
    return text


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(json.loads(sys.argv[2]))
        return 0

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--suites", nargs="+", default=["cpu", "opt", "coreml"], choices=["cpu", "opt", "coreml"])
    parser.add_argument("--quick", action="store_true", help="fewer thread counts, one timed run")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--json", type=Path, default=ML_DIR / "dist" / "reports" / "bench.json")
    args = parser.parse_args()

    import onnxruntime as ort

    models = args.models or [m for m, (file, _) in MODELS.items() if (ML_DIR / "dist" / file).exists()]
    cores = os.cpu_count() or 1
    performance_cores = cores
    if sys.platform == "darwin":
        try:
            performance_cores = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                                   text=True).stdout.strip())
        except ValueError:
            pass
    machine = {"platform": platform.platform(), "machine": platform.machine(), "cpuCount": cores,
               "performanceCores": performance_cores, "onnxruntime": ort.__version__,
               "availableProviders": ort.get_available_providers()}
    if sys.platform == "darwin":
        machine["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        machine["memoryBytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
    print(json.dumps(machine))

    thread_counts = sorted({1, 2, performance_cores, cores}) if not args.quick else sorted({1, performance_cores, cores})
    runs = 2 if args.quick else 3
    results = []
    for model_id in models:
        file, audio_seconds = MODELS[model_id]
        print(f"\n== {model_id} ({file})")
        configs = []
        if "cpu" in args.suites:
            configs += [{"suite": "cpu", "label": f"CPU, {t} thread(s)", "threads": t} for t in thread_counts]
            configs += [{"suite": "cpu", "label": "CPU, default threads", "threads": 0}]
        if "opt" in args.suites:
            configs += [{"suite": "opt", "label": f"CPU, {performance_cores} threads, optimisation {level}",
                         "threads": performance_cores, "optimization": level} for level in ("disabled", "basic", "extended")]
            if model_id != "efficientat_mn10_as":
                configs += [{"suite": "opt", "label": f"CPU, {performance_cores} threads, batch left symbolic",
                             "threads": performance_cores, "fixBatch": False}]
        if "coreml" in args.suites and "CoreMLExecutionProvider" in ort.get_available_providers():
            for model_format in (("MLProgram",) if args.quick else ("MLProgram", "NeuralNetwork")):
                for units in (("ALL",) if args.quick else ("ALL", "CPUAndGPU")):
                    configs.append({"suite": "coreml", "label": f"CoreML {model_format}, {units}",
                                    "coreml": {"ModelFormat": model_format, "MLComputeUnits": units},
                                    "verbose": True})
        for config in configs:
            config.update({"model": model_id, "file": file, "runs": runs})
            result = run_config(config, args.timeout)
            results.append(result)
            print(f"  {config['label']:<46s} {describe(result, audio_seconds)}", flush=True)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.json.read_text()) if args.json.exists() else {"results": []}
    kept = [r for r in existing["results"] if r["model"] not in models or r["suite"] not in args.suites]
    args.json.write_text(json.dumps({"machine": machine, "results": kept + results}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
