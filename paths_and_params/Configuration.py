import numpy as np
import os
from time import strftime


class Configuration():
    def __init__(self):
        self.data_path = r'C:\dev\DL_MTL-master\Data'
        self.x_dir = r'C:\dev\DL_MTL-master\Data\All'
        # self.x_dir =r'C:\dev\DL_MTL-master\Data\Segmentation\Image'
        self.y_mask = r'C:\dev\DL_MTL-master\Data\Segmentation\Mask'
        self.y_label_COVID = r'C:\dev\DL_MTL-master\Data\CT_COVID'
        self.y_label_non_COVID=r'C:\dev\DL_MTL-master\Data\CT_NonCOVID'
        self.seg_im=r'C:\dev\DL_MTL-master\Data\Segmentation\Image'
        self.save_model = r'C:\dev\DL_MTL-master\weights'
        self.save_seg_results = r'C:\dev\DL_MTL-master\Results\SegmentationResults'
        self.save_re_results = r'C:\dev\DL_MTL-master\Results\ReconstructionResults'
        self.run_dir = os.path.dirname(self.data_path) + '/weights/' + 'Run_at_time_' + strftime('%H-%M-%S') +\
                      '_date_' + strftime('%d-%m-%y')
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_10-21-17_date_10-01-22\weights_e0017_loss1.0800_val_loss1.6379_seg_loss0.0238_re_loss0.9987_class_loss0.0575_val_seg0.0952_val_re1.4400_val_clas0.1027.h5' # weights for class - vgg16 -5 on bridge (TL) and mask
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_19-13-57_date_11-01-22\weights_e0042_loss1.0129_val_loss1.0898_seg_loss0.0357_re_loss0.9468_class_loss0.0304_val_seg0.0000_val_re0.9504_val_clas0.1394.h5' # weights for class with TL and mask
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_21-33-02_date_11-01-22\weights_e0071_loss0.1408_val_loss0.2841_seg_loss0.1384_re_loss0.0024_class_loss0.0000_val_seg0.2390_val_re0.0015_val_clas0.0437.h5' # weights for segmentation (TL)
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_22-42-45_date_11-01-22/weights_e0009_loss0.0646_val_loss0.1491_seg_loss0.0238_re_loss0.0056_class_loss0.0351_val_seg0.0000_val_re0.0090_val_clas0.1402.h5' #class with TL without mask at all
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_11-09-48_date_15-01-22/weights_e0008_loss0.1946_val_loss0.2752_seg_loss0.1819_re_loss0.0127_class_loss0.0000_val_seg0.2496_val_re0.0109_val_clas0.0146.h5' # seg with vgg16 - 5 on bridge (TL and unet adjustment)

        # self.weights_path = r'C:\dev\DL_MTL-master\weights\segmentation best\weights_e0049_loss0.1127_val_loss0.2412.h5' # weights for segmentation - according to the paper
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\classification best\weights_e0002_loss0.1928_val_loss0.2466.h5' #weights for classification - according to the paper
        # self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_18-48-54_date_23-01-22\weights_e0025_loss0.3629_val_loss0.5944_seg_loss0.0476_re_loss0.3152_class_loss0.0001_val_seg0.0000_val_re0.3258_val_clas0.2685.h5' #weights for classification - according to the paper
        self.weights_path = r'C:\dev\DL_MTL-master\weights\Run_at_time_19-13-35_date_23-01-22\weights_e0030_loss0.7570_val_loss0.8210_seg_loss0.4544_re_loss0.3026_class_loss0.0000_val_seg0.5352_val_re0.2817_val_clas0.0041.h5' # weights for segmentation - according to the paper


    def createdir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path




