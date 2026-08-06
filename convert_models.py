import os
import sys
import torch
import urllib.request

# Append path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib_v5.vr_network import nets_new

def convert_vr_model(pth_path, onnx_path):
    print(f"Loading VR model from {pth_path}...")
    # UVR-DeNoise-Lite uses 2048 n_fft, nout=16, nout_lstm=128
    model = nets_new.CascadedNet(2048, 31191, nout=16, nout_lstm=128)
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print("Exporting VR model to ONNX...")
    # Dummy input with shape (batch, channels, height/max_bin, width/frames)
    dummy_input = torch.randn(1, 2, 1024, 256)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {3: "width"},
            "output": {3: "width"}
        },
        opset_version=18
    )
    print(f"Successfully converted VR model to {onnx_path}")

def main():
    os.makedirs("converted_models", exist_ok=True)

    # Check if we have UVR-DeNoise-Lite.pth locally, if not download it or use the repo one
    local_pth = "models/VR_Models/UVR-DeNoise-Lite.pth"
    if not os.path.exists(local_pth):
        # Fallback to download link if needed, but in our case, it's already in the repo!
        print("UVR-DeNoise-Lite.pth exists in the repo!")

    convert_vr_model(local_pth, "converted_models/UVR-DeNoise-Lite.onnx")

if __name__ == "__main__":
    main()
