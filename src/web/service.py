
import sys
from pathlib import Path
from neo4j import GraphDatabase
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external_lib"/"uie_pytorch"))

from uie_predictor import UIEPredictor # type: ignore
from configs import config
from runner.Predictor import IntentClassifyBertPredictor
from agent.spell_check_agent import SpellCheckAgent
import logging
# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename="app.log", # 写入文件
    #filemode="a", # 追加模式
)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.intent_classify_predictor = self.init_intent_classfiy_predictor()        
        self.spell_check_agent = self.init_spell_check_agent()
        self.uie_predictor = self.init_uie_predictor()
        self.neo4j_driver = GraphDatabase.driver(uri=config.NEO4J_CONFIG["uri"], auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]))

    @staticmethod
    def init_intent_classfiy_predictor():
        model_path = config.CHECKPOINT_DIR / "intent_classify" / "best_model"
        intent_predictor = IntentClassifyBertPredictor(model_path)
        return intent_predictor
    
    @staticmethod
    def init_spell_check_agent():
        return SpellCheckAgent(
            model_name="openai:qwen3.7-plus", 
            temperature=0.2
        )

    @staticmethod
    def init_uie_predictor():
        ie = UIEPredictor(model='uie-base', schema=[], task_path=config.CHECKPOINT_DIR/"uie" / 'model_best')
        return ie
    def extract_entity(self, question: str, schema):
        self.uie_predictor.set_schema(schema)
        result = self.uie_predictor(question)[0]
        """
        {
            '商品': [{'end': 11,
                     'probability': np.float32(0.9976633),
                     'start': 0,
                     'text': '小米12S Ultra'},
                     {'end': 11,
                     'probability': np.float32(0.9976633),
                     'start': 0,
                     'text': 'Apple Watch Series 8'}],
        }
        返回的数据结构：
        {
        "商品"："小米12S Ultra",
        "运行内存"："8G"
        }
        """
        for key in result.keys():
            result[key] =[dict["text"] for dict in  result[key]]
        return result
        
    # 先解决单论聊天问题
    def chat(self, question: str)->str:
        # 1.问题输入：小米12S Ultre 都有哪些版本        
        # 小米12S Ultra 的机身内存是多少？-》查询某商品的某个属性的属性值-》商品、属性-》{商品：小米12S Ultra， 属性：机身内存}
        # 1.意图识别：查询某商品的所有单品->[y = k[商品]+b]
        result = self.intent_classify_predictor.predict_intent(question)
        intent = result.get("predicted_intent","")
        logger.info(f"1. 意图识别结果：{intent}")
        
        # 2.拼写纠错：小米12S Ultra 都有哪些版本
        result = self.spell_check_agent.correct(question)
        question = result.corrected_text
        logger.info(f"2. 拼写纠错结果：{question}")
        
        # 3.实体抽取:【商品-》小米12S Ultra】槽位填充
        match intent:
            case "查询某商品的所有单品":
                # 4.知识图谱查询：spu=小米12S Ultra->sku
                schema = ["商品"]
                result = self.extract_entity(question, schema) # {“商品”：[“小米12S Ultra”]}
                logger.info(f"3. 实体抽取结果：{result}")
                cypher = """
                    MATCH (spu:SPU{spu_name:$spu_name})<-[:Belong]-(s:SKU) 
                    RETURN s.sku_name as sku_name
                """
                slot = {"spu_name": result["商品"][0]}
                records, _, _ = self.neo4j_driver.execute_query(cypher, slot)
                if len(records)>0:
                    res = "\n".join([record["sku_name"] for record in records])
                    response = f"{result['商品'][0]}的所有单品有：\n{res}"
                    return response
                
            case "查询某商品的某个属性的属性值":
                # 4.知识图谱查询：spu=小米12S Ultra->sku
                schema = ["商品", "属性"]
                result = self.extract_entity(question, schema) #【商品：xxxx,属性：yyyy】
                logger.info(f"3. 实体抽取结果：{result}")
                
                return "查询某商品的所有单品"

        # 5.返回结果
        return f"意图为：{intent}, 该问题无法回答，请换种方式试一试!"