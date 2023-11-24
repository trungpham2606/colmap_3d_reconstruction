import cv2
from pathlib import Path
from skimage.io import imsave

def video2image(input_video, output_dir, interval=30, image_size = 640, transpose=False):
    print(f'split video {input_video} into images ...')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % interval==0:
            h, w = image.shape[:2]
            ratio = image_size/max(h,w)
            ht, wt = int(ratio*h), int(ratio*w)
            image = cv2.resize(image,(wt,ht),interpolation=cv2.INTER_LINEAR)
            if transpose:
                v0 = cv2.getVersionMajor()
                v1 = cv2.getVersionMinor()
                if v0>=4 and v1>=5:
                    image = cv2.flip(image, 0)
                    image = cv2.flip(image, 1)
                else:
                    image = cv2.transpose(image)
                    image = cv2.flip(image, 1)

            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            imsave(f"{output_dir}/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    return count









