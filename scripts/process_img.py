import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from document_processing import Pipeline  # noqa: E402
import pprint  # noqa: E402
import argparse  # noqa: E402


def process_img(**kwargs) -> dict:
    """Runs pipeline inference on an image.

    Runs a provided document analysis pipeline on a single
    input image. Prints and returns a detailed output report.

    Args:
        img_path: Path to input image
        img_size: Resize image to this size before inference
        check_q: Whether to run pipeline quality checks
        pipeline: Configured Pipeline object

    Returns:
        dict: Pipeline output report for image
    """

    assert kwargs.get('img_path') is not None, "Missing path to image"
    img_path = kwargs.get('img_path')
    img_size = kwargs.get('img_size')
    check_q = kwargs.get('check_quality')

    pipeline = kwargs.get('pipeline')

    result = pipeline.process_img(img_path, check_quality=check_q, img_size=img_size)
    pp = pprint.PrettyPrinter(depth=4, indent=4)
    pp.pprint(result.full_report)
    return result


def main():
    parser = argparse.ArgumentParser(description='Process a single document image through the pipeline')
    parser.add_argument('-i', '--img_path', help='Image path', type=Path,
                        default=REPO_ROOT / 'samples' / 'DL_2011' / '1_CR_DL_2010.jpg', )
    parser.add_argument('-f', '--format', help='Model format', type=str,
                        choices=['ONNX', 'OpenVINO'], default='ONNX')
    parser.add_argument('-d', '--device', help='Device to run inference on', type=str,
                        choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--check_quality',
                        help='Is there need to check quality?',
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--img_size', help='To which max size reshape image', required=False, default=1500, type=int)
    args = parser.parse_args()
    params = vars(args)

    pipeline = Pipeline(model_format=params['format'], device=params['device'], )

    params['pipeline'] = pipeline
    process_img(**params)


if __name__ == '__main__':
    main()






