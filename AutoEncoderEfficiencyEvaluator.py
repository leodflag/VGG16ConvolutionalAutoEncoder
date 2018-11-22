from DataGenerator import SimpleDataGeneratorGetter, ApplicationDirPathGetter
from keras.models import model_from_json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

"""
在編碼解碼後產生圖像，存在 auto_encoder  子目錄中
"""

#  作為情節存儲
def store_as_plot(testing_data: np.ndarray, predicted_data: np.ndarray, file_path: str):
    figure, axes = plt.subplots(1, 2) #  組合許多小圖放大圖顯示

    axes[0].imshow(testing_data.reshape(224, 224, 3))  #  顯示測試資料圖片放原版  重組成224*224 3張
    axes[0].set_title('Origin')
    axes[0].get_xaxis().set_visible(False)  # 設置成不可見
    axes[0].get_yaxis().set_visible(False)

    axes[1].imshow(predicted_data.reshape(224, 224, 3))  #  顯示預測數據放AutoEncoder
    axes[1].set_title('AutoEncoder')
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    figure.savefig(file_path, bbox_inches='tight', pad_inches=0) #  保存圖片
    plt.close(figure)

    print("{} saved".format(file_path))

#  儲存成單獨的文件
def store_as_separate_file(
        testing_data: np.ndarray, path_of_testing_data: str,
        predicted_data: np.ndarray, path_of_predicted_data: str
):
    cv2.imwrite(path_of_testing_data, testing_data[0]*255)  #  保存圖像
    print("{} saved".format(path_of_testing_data))  #  格式化函數：{}不设置指定位置，按默认顺序

    cv2.imwrite(path_of_predicted_data, predicted_data[0]*255)
    print("{} saved".format(path_of_predicted_data))


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model_and_weight_storing_dir = "model_and_weight"
    #  開啟檔案
    with open(os.path.join(model_and_weight_storing_dir, "auto_encoder_model.json")) as file:
        auto_encoder = model_from_json(file.read())  #  從 JSON 資料重建模型
    # 從 HDF5 檔案載入模型參數（不含模型）  Keras 
    auto_encoder.load_weights(os.path.join(model_and_weight_storing_dir, 'auto_encoder_weight.h5'))
    print(auto_encoder.summary())  #  印出概要

    base_data_source_dir = "data_source"
    #  測試資料生成
    testing_data_generator = SimpleDataGeneratorGetter().get_generator(
        data_source_dir=os.path.join(base_data_source_dir, "testing"), batch_size=1
    ).infinitely_generate_batch_of_data_pair_tuple()  #  最初的資料生成
    #  儲存圖片檔案
    image_saving_dir = os.path.join(ApplicationDirPathGetter().execute(), "auto_encoder")
    os.makedirs(image_saving_dir, exist_ok=True)  # 創建

    for index, testing_data_pair in enumerate(testing_data_generator):  #  列舉
        testing_data = testing_data_pair[0][0].reshape(1, 224, 224, 3)
        predicted_data = auto_encoder.predict(testing_data)  # 預測測試資料 (json格式)

        store_as_plot(
            testing_data=testing_data, predicted_data=predicted_data,
            file_path=os.path.join(image_saving_dir, "{}.png".format(index))
        )
        #  儲存成單獨的文件
        # store_as_separate_file(
        #     testing_data=testing_data,
        #     path_of_testing_data=os.path.join(image_saving_dir, "{}_origin.png".format(index)),
        #     predicted_data=predicted_data,
        #     path_of_predicted_data=os.path.join(image_saving_dir, "{}.png".format(index))
        # )

    print("done")
