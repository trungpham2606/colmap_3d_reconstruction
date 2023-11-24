import argparse
from convert_video import video2image
import os
import glob
import cv2
from colmap_script import build_colmap_model_no_pose
from dataset.database import parse_database_name

def generate(config):
    output_dir_images = os.path.join(config.output, config.database_name, 'images')
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)
    # 1. split video into frames (with specified interval)
    if config.input.lower().endswith(('.mp4', '.avi')):
        video2image(config.input, output_dir_images, config.frame_inter, config.image_size, transpose=config.transpose)
    else:
        count=0
        for img_file in glob.glob(os.path.join(config.input, '*')):
            img = cv2.imread(img_file)
            if img is not None:
                cv2.imwrite(os.path.join(output_dir_images, '{}.jpg'.format(count)), img)
                count += 1
    print('[INFO] done creating images')
    # 2. generate ply file
    build_colmap_model_no_pose(parse_database_name(config.database_name), config.colmap_path, config.output)
            
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)

    # for video2image
    parser.add_argument('--input', type=str, default='example/video/mouse-ref.mp4', required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--frame_inter', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=960)
    parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)

    # for sfm
    parser.add_argument('--database_name', type=str, default='custom/obj', required=True)
    parser.add_argument('--colmap_path', type=str, 
                            default=r'F:/Downloads/COLMAP-3.8-windows-cuda/COLMAP-3.8-windows-cuda/bin/colmap.exe')
    # for sfm
    # parser.add_argument('--que_database', type=str, default='linemod/cat')
    # parser.add_argument('--que_split', type=str, default='linemod_test')
    # parser.add_argument('--ref_database', type=str, default='linemod/cat')
    # parser.add_argument('--ref_split', type=str, default='linemod_test')
    # parser.add_argument('--estimator_cfg', type=str, default='configs/gen6d_train.yaml')
    args = parser.parse_args()
    
    generate(args)
    
    
    






