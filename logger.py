import os
from options import args
import torch
from scipy import io
import time


def deletefile(filename):
    try:
        os.remove(filename)
    except BaseException:
        pass


class MyLogger:
    def __init__(self):
        pass

    def init(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.record = {
            "data": {},
            "step": {},
            "info": {},
            "args": {},
            "output": ""
        }
        #
        self.start_time = time.time()
        #
        self.filename_py = save_dir + ".pt"
        self.filename_mat = save_dir + ".mat"
        deletefile(self.filename_py)
        deletefile(self.filename_mat)
        #
        self.log_args()

    def log_args(self):
        origin_arg_dict = args.__dict__
        arg_dict = self.record["args"]
        for key in origin_arg_dict.keys():
            arg_dict[key] = str(origin_arg_dict[key])

    def log_config(self, config):
        arg_dict = self.record["args"]
        for key in config.keys():
            arg_dict[key] = str(config[key])

    def add_record(self, key, value, step):
        data_dict = self.record["data"]
        step_dict = self.record["step"]
        if type(value) is torch.Tensor:
            value = value.cpu()
            value = value.numpy()
        if not (key in data_dict.keys()):
            data_dict[key] = []
            step_dict[key] = []
        data_dict[key].append(value)
        step_dict[key].append(step)
        pass

    def add_records(self, data_dict, step):
        for key in data_dict.keys():
            self.add_record(key, data_dict[key], step)
        pass

    def addinfo(self, key, value):
        info = self.record["info"]
        info[key] = value

    def save(self):
        io.savemat(self.filename_mat, self.record, appendmat=False, do_compression=True)

    def print(self, s):
        output = self.record["output"]
        output = output + str(s) + "\n"
        print(s)

    def tik(self):
        info_dict = self.record["info"]
        self.start_time = time.time()
        info_dict["start_time"] = time.strftime("%m/%d %H:%M", time.localtime())

    def tok(self):
        info_dict = self.record["info"]
        info_dict["end_time"] = time.strftime("%m/%d %H:%M", time.localtime())
        total_time = time.time() - self.start_time
        hour = total_time // 3600
        minute = (total_time - hour * 3600) // 60
        info_dict["total_time"] = f"consumed {hour} hour {minute} minutes"


logger = MyLogger()
