import torch
from pathlib import Path
from src.feature_extractor import FeatureExtractor

def export_backbone_to_onnx(output_path: str = "models/backbone.onnx"):
    Path("models").mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Chargement de l'extracteur PyTorch...")
    model = FeatureExtractor()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    print(f"[INFO] Export du graphe computationnel vers {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["input_image"],
        output_names=["layer2_features", "layer3_features"]
    )
    print(f"[SUCCESS] Modèle ONNX sérialisé avec succès dans {output_path}")

if __name__ == "__main__":
    export_backbone_to_onnx()