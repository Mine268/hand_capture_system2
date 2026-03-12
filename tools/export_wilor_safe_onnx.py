import argparse
import os

import onnx
import torch
from onnx.external_data_helper import convert_model_to_external_data

from wilor_mini.models.wilor import WiLor


class WiLorOnnxSafe(torch.nn.Module):
    def __init__(self, model: WiLor):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.flip(dims=[-1]) / 255.0
        x = (x - self.model.IMAGE_MEAN.to(x.device, dtype=x.dtype)) / self.model.IMAGE_STD.to(x.device, dtype=x.dtype)
        x = x.permute(0, 3, 1, 2)
        batch_size = x.shape[0]

        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.model.backbone(x[:, :, :, 32:-32])
        focal_length = self.model.FOCAL_LENGTH * torch.ones(batch_size, 2, device=x.device, dtype=x.dtype)

        temp_mano_params["global_orient"] = temp_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["hand_pose"] = temp_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["betas"] = temp_mano_params["betas"].reshape(batch_size, -1)
        temp_mano_output = self.model.mano(**temp_mano_params, pose2rot=False)
        temp_vertices = temp_mano_output.vertices

        pred_mano_params = self.model.refine_net(vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length)
        mano_output = self.model.mano(**pred_mano_params, pose2rot=False)
        pred_keypoints_3d = mano_output.joints.reshape(batch_size, -1, 3)
        pred_vertices = mano_output.vertices.reshape(batch_size, -1, 3)

        global_orient_rotmat = pred_mano_params["global_orient"].reshape(batch_size, 1, 3, 3)
        hand_pose_rotmat = pred_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        betas = pred_mano_params["betas"].reshape(batch_size, -1)
        pred_cam = pred_mano_params["pred_cam"].reshape(batch_size, -1)

        return (
            pred_cam,
            global_orient_rotmat,
            hand_pose_rotmat,
            betas,
            pred_keypoints_3d,
            pred_vertices,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Export a WiLoR ONNX model that avoids rotmat_to_rotvec in the ONNX graph.")
    parser.add_argument("--wilor-root", required=True, help="Path to WiLoR-mini/wilor_mini directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to wilor_final.ckpt.")
    parser.add_argument("--output", required=True, help="Output .onnx path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--constant-folding", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    mano_model_path = os.path.join(args.wilor_root, "pretrained_models", "MANO_RIGHT.pkl")
    mano_mean_path = os.path.join(args.wilor_root, "pretrained_models", "mano_mean_params.npz")
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    output_base = os.path.basename(output_path)
    output_data_name = output_base + ".data"
    temp_path = output_path + ".tmp"

    device = torch.device(args.device)
    dtype = torch.float32

    print("Loading WiLor model...")
    model = WiLor(mano_model_path=mano_model_path, mano_mean_path=mano_mean_path, focal_length=5000, image_size=256)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"], strict=False)
    model.eval()
    model.to(device, dtype=dtype)

    export_model = WiLorOnnxSafe(model).eval().to(device, dtype=dtype)
    dummy_input = torch.randint(0, 255, (1, 256, 256, 3), dtype=dtype, device=device)

    with torch.no_grad():
        outputs = export_model(dummy_input)
        print("Output shapes:", [tuple(out.shape) for out in outputs])

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    data_path = os.path.join(output_dir, output_data_name)
    if os.path.exists(data_path):
        os.remove(data_path)

    print("Exporting ONNX...")
    torch.onnx.export(
        export_model,
        dummy_input,
        temp_path,
        input_names=["input_image"],
        output_names=[
            "pred_cam",
            "global_orient_rotmat",
            "hand_pose_rotmat",
            "betas",
            "pred_keypoints_3d",
            "pred_vertices",
        ],
        opset_version=args.opset,
        do_constant_folding=args.constant_folding,
    )

    print("Saving with external data...")
    onnx_model = onnx.load(temp_path)
    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=output_data_name,
        size_threshold=1024,
    )
    onnx.save_model(onnx_model, output_path)
    os.remove(temp_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
