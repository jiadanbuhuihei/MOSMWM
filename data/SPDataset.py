# 单人场景训练使用的是VPT数据集
# solaris给出了数据集下载的脚本

from torch.utils.data import Dataset
import json
import numpy as np
from decord import VideoReader, cpu

from . import minecraft
from .segment import Segment, SegmentId

# 该函数将action的jsonl文件转换成json文件格式然后传回通过json的格式解析
def read_actions_json(actions_path):
    try:
        with open(actions_path) as json_file:
            json_lines = json_file.readlines()
    except Exception as e_utf:
        try:
            with open(actions_path, encoding="windows-1252") as json_file:
                json_lines = json_file.readlines()
        except Exception as e_win:
            raise ValueError(
                f"Error reading file {actions_path} in utf-8 and windows-1252 encodings: {e_utf} / {e_win}"
            ) from e_win

    json_data = "[" + ",".join(json_lines) + "]"
    json_data = json.loads(json_data)
    return json_data

# 会返回指定帧和obs,act,viewmats,K,但是其中的act的最后一帧会删除，因为act是当前帧的动作，最后一帧没有动作，所以会删除
class SPDataset(Dataset):
    # 该数据集处理VPT数据，返回训练所需的参数，K, viewmats, frames, actions
    
    def __init__(
        self,
        data_dir,
        obs_resize: tuple,
    ):
        super().__init__()
        
        self.directory = data_dir
        
        with open(self.directory + "/episodes_info.json", "r") as json_file:
            self.episodes_info = json.load(json_file)
            
        self._num_episodes = self.episodes_info["num_episodes"]
        self._lengths = np.array([
            ep["length"] for ep in self.episodes_info["episodes"]
        ])
        
        self._obs_resize = obs_resize
        
        self.K = minecraft.get_K(*obs_resize)
        
    def lengths(self):
        return self._lengths
    
    def num_episodes(self):
        return self._num_episodes
    
    def __getitem__(self, segment_id): # SegmentId (episode_id, start, end)
        episodes_info = self.episodes_info["episodes"][segment_id.episode_id]
        video_path = self.directory + "/" + episodes_info["video_path"]
        actions_path = self.directory + "/" + episodes_info["actions_path"]

        try:
            decord_video = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}") from e
        
        try:
            # 得到act hot map 一个二维矩阵，时间步数 x 动作维度
            act = self.read_act_slice(actions_path, segment_id.start, segment_id.stop)
        except Exception as e:
            raise ValueError(
                f"Error reading episode actions {segment_id.episode_id}: {e}"
            ) from e

        try:
            # 得到对应帧（resized后的）
            obs_decord = minecraft.read_obs_slice_decord(
                decord_video,
                segment_id.start,
                segment_id.stop,
                self._obs_resize,
            )

        except Exception as e:
            raise e
        
        # 对act中的xyz,yaw,pitch来求得相机外参 c2w
        viewmats = minecraft.act_to_cameras(act)
        viewmats = viewmats[0::4]
        
        segment = Segment(obs_decord, act[:-1], viewmats, self.K)
        
        return segment
        
    def read_act_slice(self, path, start, end):
        full_actions_json = read_actions_json(path)
        return minecraft.read_act_slice_vpt(
            full_actions_json,
            start,
            end,
        )
        
        
if __name__ == "__main__":
    data_dir = "/public_datasets/qjl/world_model/dataset/test"
    dataset_test = SPDataset(data_dir, (704, 1280))
    print(dataset_test.lengths())
    segment_id = SegmentId(episode_id=0,start=0,stop=10)
    segment = dataset_test[segment_id]
    print(segment.obs.shape, segment.act.shape, segment.viewmats.shape, segment.K.shape)
    