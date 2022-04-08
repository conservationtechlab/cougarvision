from tqdm import tqdm
from PIL import Image, ImageOps
from typing import Any, Iterable, Mapping, Sequence

def load_to_crop(img_path: str,
                  bbox_dicts: Iterable[Mapping[str, Any]],
                  confidence_threshold: float
):


    did_download = False
    num_new_crops = 0
    crops = []
    # crop_path => normalized bbox coordinates [xmin, ymin, width, height]
    bboxes_tocrop: dict[str, list[float]] = {}
    for i, bbox_dict in enumerate(bbox_dicts):
        # only ground-truth bboxes do not have a "confidence" value
        if 'conf' in bbox_dict and bbox_dict['conf'] < confidence_threshold:

            bbox_dicts.pop(i)
            continue
        # if bbox_dict['category'] != 'animal':
        #     continue
    if len(bboxes_tocrop) == 0:
        return did_download, num_new_crops

    img = Image.open(img_path)
    img.show()

    assert img is not None, 'image failed to load or download properly'
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')  # always save as RGB for consistency

    
    # crop the image
    for crop_path, bbox in bboxes_tocrop.items():
        num_new_crops += 1
        crops.append([crop(
            img, bbox_norm=bbox), bbox])

    return did_download, num_new_crops, crops



def crop(img: Image.Image, bbox_norm: Sequence[float]) -> bool:
    """Crops and returns an image

    Args:
        img: PIL.Image.Image object, already loaded
        bbox_norm: list or tuple of float, [xmin, ymin, width, height] all in
            normalized coordinates

    Returns: cropped image
    """
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if box_w == 0 or box_h == 0:
        tqdm.write(f'Skipping size-0 crop (w={box_w}, h={box_h})')
        return False

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
    
    return crop
