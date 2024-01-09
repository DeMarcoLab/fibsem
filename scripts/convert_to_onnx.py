from fibsem.segmentation.onnx_model import export_model_to_onnx
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to openfibsem model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to save onnx model",
        required=True,
    )
    args = parser.parse_args()
    export_model_to_onnx(args.checkpoint, args.output)




if __name__ == "__main__":
    main()