import os

class config:
    root_path = os.path.dirname(os.path.abspath(__file__))
    device = "cuda"

    # data config
    raw_data_path = os.path.join(root_path, "raw_data")
    

    # model config
    # ppcnet
    gaussian_blur_kernel = 0 # odd number
    # wptnet

    # loss config


    # optimizer config
    learning_rate = 5e-4

    # train config
    batch_size = 32
    train_epochs = 10
    
    pass