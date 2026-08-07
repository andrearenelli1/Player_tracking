"""
Trimmed down to only the model actually used by this fork (HRNet/WASB).
The original repo (https://github.com/nttcom/WASB-SBDT) supports several
more architectures (TrackNetV2, MonoTrack, ResTrackNetV2, DeepBall,
BallSeg) -- see its history if you need one of those back.
"""
from .hrnet import HRNet

__factory = {
    'hrnet': HRNet,
}


def build_model(cfg):
    model_name = cfg['model']['name']
    if model_name not in __factory.keys():
        raise KeyError('invalid model: {}'.format(model_name))
    return __factory[model_name](cfg['model'])
