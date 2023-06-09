import math

class Args(object):
    def __init__(self):
        self.synthetic_train_data_dir = '/content/aster_test/data/Case-Sensitive-Scene-Text-Recognition-Datasets-master/iiit5k_train/'
        self.test_data_dir = '/content/aster_test/data/Case-Sensitive-Scene-Text-Recognition-Datasets-master/iiit5k_test/'
        #self.test_data_dir = '/data/Case-Sensitive-Scene-Text-Recognition-Datasets-master/cute80_test/'
        self.logs_dir = '/logs'
        self.real_logs_dir = '/logs'
        self.vis_dir = 'iiit5k_test_visualization'
        #self.vis_dir = 'visualization'
        self.batch_size = 128
        self.workers = 0
        self.height = 64
        self.width = 256
        self.keep_ratio = False  #保持图片原有比例
        self.voc_type = 'ALLCASES_SYMBOLS'
        self.num_train = math.inf
        self.num_test = math.inf
        self.image_path = '/data/demo.png'
        self.tps_inputsize = [32, 64]
        self.tps_outputsize = [32,100]
        self.arch = 'ResNet_ASTER'
        self.max_len = 100
        self.n_group = 1
        self.STN_ON = True
        self.stn_activation = None
        self.stn_with_dropout = False
        self.loss_weights = [1.0,1.0,1.0]
        self.seed = 1
        self.print_freq = 100

        self.evaluation_metric = 'accuracy'
        self.evaluate_with_lexicon = False
        self.debug = False
        self.run_on_remote = False

        #不确定超参
        self.decoder_sdim = 512
        self.attDim = 512
        self.grad_clip = 1.0

        #超参
        self.lr = 1.0
        self.dropout = 0.5
        self.tps_margins = [0.05,0.05]
        self.num_control_points = 20
        self.with_lstm = True
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.epochs = 2000
        self.beam_width = 5

        #  训练
        self.evaluate = False
        self.resume = ''
        #  测试
        # self.evaluate = True
        # self.beam_width = 5
        # self.resume = 'C://aster.pytorch-master//demo.pth.tar'

        # colab
        self.cuda = True
        # 本地
        #self.cuda = False

