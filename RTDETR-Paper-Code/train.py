import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/LDFA-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='hit-uav/dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=4,
                # device='0,1', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                # resume='', # last.pt path
                patience=0,
                project='runs/train',
                name='exp',
                )