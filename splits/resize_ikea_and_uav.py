'''
IKEA-FA and UAV datasets are of larger resolution than the required 224x224 for training. To reduce dataloading time during training, we save downsampled versions of the dataset first.
'''
import cv2
import imutils
import os
import numpy as np
import datetime
from multiprocessing import Pool


def extract_frames(path, replace_func):
    # assert os.path.exists(new_path), f'{new_path} not exist'
    # assert len(os.listdir(new_path)) > 0, f'{new_path} empty'

    cap = cv2.VideoCapture(path)
    new_path = replace_func(path)
    if os.path.exists(new_path):
        return
    os.makedirs(new_path, exist_ok=True)
    counter = 1
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: #done
            break
        h, w, c = img.shape
        # short side scale to 224
        if h >= w:
            resized_img = imutils.resize(img, width=224)
        else:
            resized_img = imutils.resize(img, height=224)
        # print(os.path.join(new_path, f"{counter:05d}.jpg"))
        cv2.imwrite(os.path.join(new_path, f"{counter:05d}.jpg"), resized_img)
        counter += 1
    cap.release()

def main(ind):
    '''
    Fill in your own base_dir and np_path for where your UAV and Ikea datasets are located"
    '''
    # base_dir = "/dccstor/lwll_data/data/uav/videos/"
    # np_path = "/dccstor/lwll_data/omnivore/splits/uav/val.npy"

    base_dir = "/dccstor/lwll_data/data/ikea_furniture/videos/"
    np_path = "/dccstor/lwll_data/omnivore/splits/ikea_furniture/val.npy"

    uav_list = np.load(np_path)[ind::WORLD_SIZE] #extract subset
    replace_func = lambda x: x.replace('videos/', 'frames/').replace('.avi','').replace('.MP4','')
    print(f"Start ({ind}/{WORLD_SIZE})", len(uav_list), np_path)
    for i, img_path in enumerate(uav_list):
        if i % 50 == 0:
            print(f"({ind}/{WORLD_SIZE}) {datetime.datetime.now()}: {i}/{len(uav_list)}")

        extract_frames(os.path.join(base_dir, img_path), replace_func)
    return f"Done {ind}/{WORLD_SIZE}"

if __name__ == '__main__':
    WORLD_SIZE = 16
    with Pool(WORLD_SIZE) as p:
        print(p.map(main, list(range(WORLD_SIZE))
                    ))

