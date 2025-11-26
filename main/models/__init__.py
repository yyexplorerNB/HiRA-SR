import importlib
from copy import deepcopy
from os import path as osp
import sys ; sys.path.append('./')
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY



# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
try:
    _model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]
except:
    _model_modules = [importlib.import_module(f'main.models.{file_name}') for file_name in model_filenames]