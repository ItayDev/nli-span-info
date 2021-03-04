from enum import Enum, auto
from datasets import concatenate_datasets
import torch


class MODE(Enum):
    TRAIN_ALL = auto()
    TRAIN_SNLI = auto()
    TRAIN_MNLI = auto()
    TEST_SNLI = auto()
    TEST_MNLI_MATCHED = auto()
    TEST_MNLI_MISMATCHED = auto()


class TrainLoaderFactory:
    def __init__(self, batch_size, snli, mnli, collate_fn=None):
        self.batch_size = batch_size
        self.snli = snli
        self.mnli = mnli
        self.collate_fn = collate_fn
        self.loader_options = self.__build_loader_options()

    def get_loader(self, mode):
        return self.loader_options[mode]()

    def __get_all_train(self):
        all_data = concatenate_datasets([self.snli['train'], self.mnli['train']])
        return self.__get_data_loader(all_data)

    def __get_snli_train(self):
        return self.__get_data_loader(self.snli['train'])

    def __get_mnli_train(self):
        return self.__get_data_loader(self.mnli['train'])

    def __get_snli_test(self):
        return self.__get_data_loader(self.snli['test'])

    def __get_test_matched(self):
        return self.__get_data_loader(self.mnli['validation_matched'])

    def __get_test_mismatched(self):
        return self.__get_data_loader(self.mnli['validation_mismatched'])

    def __build_loader_options(self):
        return {
            MODE.TRAIN_ALL: self.__get_all_train,
            MODE.TRAIN_SNLI: self.__get_snli_train,
            MODE.TRAIN_MNLI: self.__get_mnli_train,
            MODE.TEST_SNLI: self.__get_snli_test,
            MODE.TEST_MNLI_MATCHED: self.__get_test_matched,
            MODE.TEST_MNLI_MISMATCHED: self.__get_test_mismatched,
        }

    def __get_data_loader(self, data):
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           shuffle=True)