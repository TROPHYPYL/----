####################处理完后可以删除###########################
import sys
from pathlib import Path
from turtle import mode
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT/"external_lib" / "uie_pytorch"))
from external_lib.uie_pytorch.uie_predictor import SchemaTree
from uie_predictor import UIEPredictor
from pprint import pp, pprint
# schema = ['时间','选手','赛事名称']
# ie = UIEPredictor(model = 'uie-base',schema = schema)
# pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌和赵国弼分别以188.25分获得金牌！"))

from src.configs import config
from uie_predictor import UIEPredictor
schema = ["商品", "颜色"]
ie = UIEPredictor( model="uie-base", schema=schema,task_path=config.CHECKPOINT_DIR/"uie"/"model_best")
pprint(ie( "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充8GB+128GB 冷杉绿 5G手机"))
