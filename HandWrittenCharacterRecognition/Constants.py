
class Constants:
    DATA_SET_PATH = 'Dataset/'
    params_cnn_1 = {}
    params_cnn_1["1"] = [6, 16, 120, 84]
    params_cnn_1["2"] = [18, 32, 256, 84]
    params_cnn_1["3"] = [18, 32, 128, 0]

    params_cnn_2 = {}
    params_cnn_2["1"] = [32, 128, 0]
    params_cnn_2["2"] = [64, 256, 0]
    params_cnn_2["3"] = [128, 512, 256]

    params_cnn_3 = {}
    params_cnn_3["1"] = [32, 32, 64, 64, 0, 512]
    params_cnn_3["2"] = [16, 16, 32, 64, 0, 128]
    params_cnn_3["3"] = [32, 32, 64, 64, 256, 128]