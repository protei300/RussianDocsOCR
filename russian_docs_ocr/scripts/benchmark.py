import sys
sys.path.append('..')
from document_processing import Pipeline
from pathlib import Path
import argparse
import json
import cpuinfo
import pandas as pd
import numpy as np

def benchmark(**kwargs):
    """Runs pipeline benchmark on images and saves results.

    Runs pipeline multiple times on given images to benchmark
    performance. Saves execution time by device and document type
    to a CSV file.

    Args:
        kwargs: Dictionary of benchmark parameters
            (format, device, images folder, save location, etc.)

    """
    def test_pic(img_folder: Path):
        """Tests pipeline speed on given image folder.

        Runs pipeline on all images in folder repeatedly.
        Calculates average execution time.

        Args:
            img_folder (Path): Folder with test images

        Returns:
            float: Average run time
        """
        benchmark_list = []

        for _ in range(kwargs['cicles']):
            for img in img_folder.glob('**/*.*'):
                if img.suffix == '.json':
                    continue
                result = pipeline(img)
                benchmark_list.append(result.timings['total'])

        return np.mean(benchmark_list)


    pipeline = Pipeline(model_format=kwargs['format'], device=kwargs['device'])


    images_folder = Path(kwargs['images'])
    benchmark_folder = Path(kwargs['save_to'])
    benchmark_folder = benchmark_folder.joinpath(f"{kwargs['format']}_{kwargs['device']}.csv")
    doctypes = [folder for folder in images_folder.iterdir() if folder.is_dir()]

    #preheat model
    pipeline(next(iter(images_folder.glob('**/*.jpg'))))


    if benchmark_folder.is_file():
        df_result = pd.read_csv(benchmark_folder, index_col=0)
    else:
        raw_doctypes = list(map(lambda x: x.stem, doctypes))
        df_result = pd.DataFrame(columns=raw_doctypes)


    if kwargs['device'] == 'cpu':
        device_name = cpuinfo.get_cpu_info()['brand_raw']
    else:
        import subprocess
        n = str(subprocess.check_output(["nvidia-smi", "-L"]))
        device_name = n.split('GPU 0: ')[1].split(' (')[0]


    bench_result = {device_name: {}}


    if len(doctypes) > 0:
        for img_folder in doctypes:
            print(f'[*] Collecting info for {img_folder.name}')
            result = test_pic(img_folder)
            bench_result[device_name][img_folder.stem] = result
    else:
        result = test_pic(images_folder)
        bench_result[device_name]['all'] = result

    benchmark_folder.parent.mkdir(parents=True, exist_ok=True)
    df_result = pd.concat((df_result, pd.DataFrame.from_dict(bench_result, orient='index')), ignore_index=False, axis=0)
    df_result = df_result.groupby(df_result.index).mean()
    df_result = df_result.round(3)
    df_result.to_csv(benchmark_folder)

def main():
    parser = argparse.ArgumentParser(description='Benchmark pipeline')
    parser.add_argument('-i', '--images', help='From where to read images', type=str, default='../samples')
    parser.add_argument('-s', '--save_to', help='Where to save result in JSON format', type=str,
                        default=r'bench_results')
    parser.add_argument('-f', '--format', help='Select model format TFlite, ONNX, OpenVINO', type=str,
                        default='ONNX')
    parser.add_argument('-d', '--device', help='On which device to run - cpu or gpu', default='cpu', type=str)
    parser.add_argument('--img_size', help='To which max size reshape image', required=False, default=1500, type=int)
    parser.add_argument(
        '--cicles',
        help='How many cicles to run in images, more better accuracy',
        required=False,
        default=1,
        type=int)


    args = parser.parse_args()
    params = vars(args)

    benchmark(**params)


if __name__ == '__main__':
    main()
