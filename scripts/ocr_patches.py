import sys
sys.path.append('..')
from document_processing import Pipeline
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm
import os
from shutil import rmtree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_patches(*args, **kwargs):
    """Extracts and saves words patches from documents.

    Runs a document pipeline on images to detect words.
    Saves extracted word patches organized by document type and
    text field to prepare OCR training data.

    Args:
       folder_to: Folder path for saving patches
       folder_from: Folder path containing document images
       clear_folder: Whether to clear folder_to before saving

    """

    folder_to = kwargs.get('folder_to')
    assert folder_to is not None, 'Missing folder to save'
    folder_to = Path(folder_to)

    clear_folder = kwargs.get('clear_folder', False)
    if clear_folder:
        rmtree(folder_to, ignore_errors=True)

    folder_from = kwargs.get('folder_from')
    assert folder_from is not None, 'Missing img path'
    folder_from = Path(folder_from)



    pipeline = Pipeline()


    for img_path in tqdm(folder_from.glob('**/*.*')):
        if img_path.suffix not in ['.jpg', '.png', '.jpeg']:
            continue
        result = pipeline(img_path,
                          ocr=True,
                          check_quality=False,
                          low_quality=True,
                          img_size=1500)
        doctype = result.doctype
        patches = result.words_patches
        if not patches:
            continue

        for field_name, patch in patches.items():
            img_path = folder_to.joinpath(doctype, field_name)
            img_path.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(patch['patches']):
                ocr_res = patch['ocr'][i]
                for symb in r':/\*':
                    ocr_res = ocr_res.replace(symb, '')

                # if nothing detected
                if len(ocr_res) == 0:
                    continue

                img_file = img_path.joinpath(ocr_res + '.jpg')
                j = 0
                while img_file.exists():
                    img_file = img_file.with_stem(
                        img_file.stem.split('[', maxsplit=1)[0] + f'[{j}]'
                    )
                    j += 1
                try:
                    Image.fromarray(img).save(img_file)
                except:
                    print(f"failed to write file {img_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Make patches for OCR')
    parser.add_argument('-t', '--folder_to', help='Where to save results', required=True, type=str)
    parser.add_argument('-f', '--folder_from', help='Where to read images', required=True, type=str)
    parser.add_argument('--img_size', help='To which max size reshape image', required=False, default=1500, type=int)
    parser.add_argument('-clr', '--clear_folder', action=argparse.BooleanOptionalAction, default=False,
                        help="Should we clear folder_to?")

    args = parser.parse_args()

    params = vars(args)
    get_patches(**params)



if __name__ == '__main__':
    main()



