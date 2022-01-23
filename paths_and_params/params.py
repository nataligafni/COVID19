class Params():
    def __init__(self):

        self.data_type = 'seg_only' #'seg_only' #'not_seg' # 'all'

        self.feature_maps = 64
        self.kernel = 3
        self.pool_size = (2, 2)
        self.up_sample_size = (2, 2)

        self.dim = (256, 256, 3)

        self.loss_weights = [1,1,1]

        self.drop_out = 0.0
        self.batch_size = 6
        self.epochs = 100
        self.lr = 1e-4

        self.seg_thresh = 0.5
        self.class_thresh = 0.5

        self.train_flag = False