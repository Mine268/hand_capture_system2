import argparse
import os

import onnx
import torch
from onnx.external_data_helper import convert_model_to_external_data

from wilor_mini.models.mano_wrapper import MANO
from wilor_mini.models.wilor import WiLor


class WiLorBackboneParams(torch.nn.Module):
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
        global_orient_rotmat = pred_mano_params["global_orient"].reshape(batch_size, 1, 3, 3)
        hand_pose_rotmat = pred_mano_params["hand_pose"].reshape(batch_size, 15, 3, 3)
        betas = pred_mano_params["betas"].reshape(batch_size, 10)
        pred_cam = pred_mano_params["pred_cam"].reshape(batch_size, 3)

        return pred_cam, global_orient_rotmat, hand_pose_rotmat, betas


class ManoCpuModel(torch.nn.Module):
    def __init__(self, mano_model_path: str):
        super().__init__()
        self.mano = MANO(model_path=mano_model_path, create_body_pose=False)

    def forward(self, global_orient_rotmat, hand_pose_rotmat, betas):
        mano_output = self.mano(
            global_orient=global_orient_rotmat,
            hand_pose=hand_pose_rotmat,
            betas=betas,
            pose2rot=False,
        )
        pred_keypoints_3d = mano_output.joints.reshape(global_orient_rotmat.shape[0], -1, 3)
        pred_vertices = mano_output.vertices.reshape(global_orient_rotmat.shape[0], -1, 3)
        return pred_keypoints_3d, pred_vertices


def export_with_external_data(model, dummy_inputs, output_path, output_names, opset):
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    output_base = os.path.basename(output_path)
    output_data_name = output_base + ".data"
    temp_path = output_path + ".tmp"

    os.makedirs(output_dir, exist_ok=True)
    for path in [temp_path, output_path, os.path.join(output_dir, output_data_name)]:
        if os.path.exists(path):
            os.remove(path)

    torch.onnx.export(
        model,
        dummy_inputs,
        temp_path,
        input_names=[inp for inp, _ in dummy_inputs] if isinstance(dummy_inputs, list) else None,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=False,
    )

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


def parse_args():
    parser = argparse.ArgumentParser(description="Export split WiLoR backbone and MANO ONNX models.")
    parser.add_argument("--wilor-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-output", required=True)
    parser.add_argument("--mano-output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    mano_model_path = os.path.join(args.wilor_root, "pretrained_models", "MANO_RIGHT.pkl")
    mano_mean_path = os.path.join(args.wilor_root, "pretrained_models", "mano_mean_params.npz")

    device = torch.device(args.device)
    dtype = torch.float32

    print("Loading WiLor...")
    wilor = WiLor(mano_model_path=mano_model_path, mano_mean_path=mano_mean_path, focal_length=5000, image_size=256)
    wilor.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"], strict=False)
    wilor.eval().to(device, dtype=dtype)

    backbone_model = WiLorBackboneParams(wilor).eval().to(device, dtype=dtype)
    dummy_image = torch.randint(0, 255, (1, 256, 256, 3), dtype=dtype, device=device)
    with torch.no_grad():
        backbone_outputs = backbone_model(dummy_image)
        print("Backbone output shapes:", [tuple(t.shape) for t in backbone_outputs])

    print("Exporting backbone model...")
    torch.onnx.export(
        backbone_model,
        dummy_image,
        args.backbone_output + ".tmp",
        input_names=["input_image"],
        output_names=["pred_cam", "global_orient_rotmat", "hand_pose_rotmat", "betas"],
        opset_version=args.opset,
        do_constant_folding=False,
    )
    backbone_onnx = onnx.load(args.backbone_output + ".tmp")
    convert_model_to_external_data(
        backbone_onnx,
        all_tensors_to_one_file=True,
        location=os.path.basename(args.backbone_output) + ".data",
        size_threshold=1024,
    )
    onnx.save_model(backbone_onnx, args.backbone_output)
    os.remove(args.backbone_output + ".tmp")
    print("Saved:", args.backbone_output)

    print("Loading MANO CPU wrapper...")
    mano_cpu_model = ManoCpuModel(mano_model_path).eval()
    global_orient = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
    hand_pose = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, 15, 1, 1)
    betas = torch.zeros((1, 10), dtype=torch.float32)
    with torch.no_grad():
        mano_outputs = mano_cpu_model(global_orient, hand_pose, betas)
        print("MANO output shapes:", [tuple(t.shape) for t in mano_outputs])

    print("Exporting MANO CPU model...")
    torch.onnx.export(
        mano_cpu_model,
        (global_orient, hand_pose, betas),
        args.mano_output + ".tmp",
        input_names=["global_orient_rotmat", "hand_pose_rotmat", "betas"],
        output_names=["pred_keypoints_3d", "pred_vertices"],
        opset_version=args.opset,
        do_constant_folding=False,
    )
    mano_onnx = onnx.load(args.mano_output + ".tmp")
    convert_model_to_external_data(
        mano_onnx,
        all_tensors_to_one_file=True,
        location=os.path.basename(args.mano_output) + ".data",
        size_threshold=1024,
    )
    onnx.save_model(mano_onnx, args.mano_output)
    os.remove(args.mano_output + ".tmp")
    print("Saved:", args.mano_output)


if __name__ == "__main__":
    main()
