import time
import numpy as np
import torch
import onnxruntime as ort
from src.feature_extractor import FeatureExtractor

def benchmark_inference(num_warmup: int = 20, num_iterations: int = 100):
    print("=======================================================")
    print("      BENCHMARK DE LATENCE INDUSTRIELLE (CPU)")
    print("=======================================================")

    # 1. Trame d'entrée normalisée (1 image RGB 224x224)
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_np)

    # ---------------------------------------------------------
    # Test 1 : PyTorch Natif (CPU)
    # ---------------------------------------------------------
    print("\n[1/2] Évaluation du runtime PyTorch...")
    pytorch_model = FeatureExtractor()
    pytorch_model.eval()

    # Warm-up (préchauffage du cache d'instructions CPU)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = pytorch_model(dummy_tensor)

    # Mesure temporelle
    start_pt = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = pytorch_model(dummy_tensor)
    end_pt = time.perf_counter()

    pt_total_time = (end_pt - start_pt) * 1000.0  # en ms
    pt_latency = pt_total_time / num_iterations
    pt_fps = 1000.0 / pt_latency

    # ---------------------------------------------------------
    # Test 2 : ONNX Runtime (CPU avec optimisations SIMD)
    # ---------------------------------------------------------
    print("[2/2] Évaluation du runtime ONNX Runtime...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession("models/backbone.onnx", session_options, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy_np})

    # Mesure temporelle
    start_ort = time.perf_counter()
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: dummy_np})
    end_ort = time.perf_counter()

    ort_total_time = (end_ort - start_ort) * 1000.0
    ort_latency = ort_total_time / num_iterations
    ort_fps = 1000.0 / ort_latency

    speedup = ((pt_latency - ort_latency) / pt_latency) * 100.0

    print("\n================ RÉSULTATS DU BENCHMARK ================")
    print(f"• PyTorch CPU Latency     : {pt_latency:.2f} ms / frame  ({pt_fps:.1f} FPS)")
    print(f"• ONNX Runtime Latency    : {ort_latency:.2f} ms / frame  ({ort_fps:.1f} FPS)")
    print(f"• Gain de latence         : {speedup:+.1f} %")
    print("========================================================")

if __name__ == "__main__":
    benchmark_inference()