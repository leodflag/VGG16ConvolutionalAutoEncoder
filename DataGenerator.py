import os
import sys
import numpy as np
import cv2   #  opencv
import math
import itertools  #  迭代器
"""
取得檔案、生產資料的自建函數
"""
#  隨機化折疊數據
class RandomizeKFoldDataGeneratorPairGenerator:
    #  生成資料(生成器函數)：將訓練與檢測湊一對
    def get_training_and_validation_data_generator_pair(
            self, num_of_folds: int, data_source_dir: str,
            batch_size_of_training_data_set: int, batch_size_of_validation_data_set: int
    ):
        all_data_files = np.array(DataFileListGetter().execute(data_source_dir))
        validation_data_files_within_every_fold = self.__get_split_data_files(all_data_files, num_of_folds)

        for validation_data_files in validation_data_files_within_every_fold:
            #  setdiff1d(X,Y)  集合的差，即元素在X中且不在Y中  全部跟檢測集合的差
            training_data_files = np.setdiff1d(all_data_files, validation_data_files)
            training_data_generator = DataGenerator(training_data_files, batch_size_of_training_data_set)

            validation_data_generator = DataGenerator(validation_data_files, batch_size_of_validation_data_set)

            print("training: {}, testing: {}".format(training_data_files.size, validation_data_files.size))
            # 生成器函數用yield當return
            yield training_data_generator, validation_data_generator

    #  靜態方法  隨機取得切分後的資料
    @staticmethod
    def __get_split_data_files(data_files, num_of_folds):
        #  floor 取得數字的下界
        size_of_data_within_single_fold = int(math.floor(data_files.size / num_of_folds))
        temp_data_files = data_files  #  板模資料檔案
        split_data_files = list()
        #  隨機選取
        for _ in range(num_of_folds - 1):
            selected_data = np.random.choice(temp_data_files, size=size_of_data_within_single_fold, replace=False)
            split_data_files.append(selected_data)  #  隨機選取的檔案
            temp_data_files = np.setdiff1d(temp_data_files, selected_data)  # 兩者集合差

        split_data_files.append(temp_data_files)

        return split_data_files

#  簡單的資料生成  return類別DataGenerator，可用方法
class SimpleDataGeneratorGetter:
    @staticmethod
    def get_generator(data_source_dir, batch_size):
        data_file_list = DataFileListGetter().execute(data_source_dir)  #  執行
        return DataGenerator(data_file_list, batch_size)


class DataGenerator:
    def __init__(self, data_file_list, batch_size):
        self.__data_file_list = data_file_list
        self.__batch_size = batch_size
    # 無限生成批量
    def infinitely_generate_batch_of_data_pair_tuple(self):
        file_to_convert = list()
        for file in itertools.cycle(self.__data_file_list):  #  cycle  对()中的元素反复执行循环，返回迭代器
            if len(file_to_convert) < self.__batch_size:
                file_to_convert.append(file)
            else:
                pre_processed_data = DataConverter().read_data_then_expand_and_standardize(file_to_convert)
                yield pre_processed_data, pre_processed_data
                file_to_convert = list()
    #  逐量生成數據組
    def generate_batch_of_data_pair_tuple(self):
        for start_index in range(0, len(self.__data_file_list), self.__batch_size):
            end_index = start_index + self.__batch_size  #  移動窗格
            batch_of_data_file_list = self.__data_file_list[start_index:end_index]
            #  將選取的窗格展開並標準化
            pre_processed_data = DataConverter().read_data_then_expand_and_standardize(batch_of_data_file_list)

            yield pre_processed_data, pre_processed_data


class DataFileListGetter:
    #  打開檔案
    @staticmethod
    def execute(data_source_dir: str) -> list:
        return [
            os.path.join(data_source_dir, file)
            for file in os.listdir(data_source_dir)
            if os.path.isfile(
                os.path.join(data_source_dir, file)
            )
        ]

#  數據轉換
class DataConverter:
    #  讀取數據擴展和標準化
    def read_data_then_expand_and_standardize(self, data_file_list: list) -> np.ndarray:
        data_list = self.__fetch_all_data_from_disk(data_file_list)  #  從磁碟獲取所有數據
        converted_data_list = self.__expand_and_standardize(data_list)  #  展開並標準化
        return converted_data_list
    #  從磁碟路徑取得所有資料
    @staticmethod
    def __fetch_all_data_from_disk(data_file_list: list) -> np.ndarray:
        data_list = list()

        for data_file in data_file_list:
            data_list.append(
                np.array(
                    cv2.imread(data_file) # 讀取檔案
                )
            )

        return np.array(data_list)
    #  展開和規範
    @staticmethod
    def __expand_and_standardize(data_list: np.ndarray) -> np.ndarray:  #ndarray 多重額度
        converted_data_list = data_list.astype(np.float32)  #  數據轉換類型
        converted_data_list = converted_data_list/255  #  標準化

        return converted_data_list

#  應用程序檔案路徑取得
class ApplicationDirPathGetter:
    @staticmethod
    def execute() -> str:  # -> 提示该函数 输入参数 和 返回值 的数据类型  方便閱讀程式碼
        if getattr(sys, 'frozen', False):  #  返回對象屬性
            application_path = sys.executable #  可執行

        elif hasattr(sys.modules['__main__'], "__file__"): # 判断对象是否包含对应的属性
            application_path = os.path.abspath(sys.modules['__main__'].__file__)

        else:
            application_path = sys.executable
            
        return os.path.dirname(application_path)

