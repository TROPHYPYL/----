
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
from repository.mysql_repository import MySQLRepository
from repository.chroma_repository import ChromaRepository
from dialogue.intent_rule import match_intent
from dialogue.rewriter import QuestionRewriter
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
        # self.uie_predictor = self.init_uie_predictor()
        # self.neo4j_driver = GraphDatabase.driver(uri=config.NEO4J_CONFIG["uri"], auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]))

        # self.mysql_repository = MySQLRepository()
        # self.chroma_repository = ChromaRepository()
        
        self.rewriter = QuestionRewriter()
    @staticmethod
    def init_intent_classfiy_predictor():
        model_path = config.CHECKPOINT_DIR / "intent_classify" / "best_model"
        intent_predictor = IntentClassifyBertPredictor(model_path)
        return intent_predictor
    
    @staticmethod
    def init_spell_check_agent():
        return SpellCheckAgent(
            model_name="openai:qwen3.7-plus-2026-05-26", 
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
    
    # 属性抽取--》{属性：机身内存}
    def extract_attr(self, question):
        """用户config.SCHEMA抽取属性名，取最长匹配"""
        self.uie_predictor.set_schema(config.SCHEMA)
        found = {k for k, v in self.uie_predictor(question)[0].items() if v}
        return max(found, key=len, default="")
    
    def entity_align(self, entity_type, entity):
        standard_word_mysql = self.mysql_repository.get_standard_word(entity_type, entity)
        if standard_word_mysql:
            logger.info(f"同义词表查询结果：{standard_word_mysql}")
            return standard_word_mysql
        else:
            standard_word_chroma, distance = self.chroma_repository.get_standard_word_by_synonym(entity_type, entity)
            logger.info(f"向量库查询结果：{standard_word_chroma}, 距离：{distance}")
            if standard_word_chroma and distance < 0.5:
                # 免审核可以直接使用，并且需要添加同义词表
                self.mysql_repository.add_synonym(
                    entity_type=entity_type,
                    std_name=standard_word_chroma,
                    synonym=entity,
                    distance=distance, 
                    is_reviewed=1
                )
                
                return standard_word_chroma
            else:
                # 需要审核
                self.mysql_repository.add_synonym(
                                    entity_type=entity_type,
                                    std_name=standard_word_chroma,
                                    synonym=entity,
                                    distance=distance, 
                                    is_reviewed=0
                                )
                return None
    # 先解决单论聊天问题
    def chat(self, question: str)->str:
        # 1.问题输入：小米12S Ultre 都有哪些版本        
        # 小米12S Ultra 的机身内存是多少？-》查询某商品的某个属性的属性值-》商品、属性-》{商品：小米12S Ultra， 属性：机身内存}
        # logger.info(f"聊天历史记录：{session}")
        # # 0.问题重述：指代消解 
        # question = self.rewriter.rewrite(question, session.history)
        # logger.info(f"0.问题重述结果：{question}")
        
        # 1.意图识别：查询某商品的所有单品->[y = k[商品]+b]
        #   1.1 规则判断
        intent = match_intent(question)
        logger.info(f"1.1. 规则获取意图识别结果：{intent}")
        if intent is None:
        #   1.2 模型判断
            result = self.intent_classify_predictor.predict_intent(question)
            intent = result.get("predicted_intent","")
            logger.info(f"1.2 Bert模型获取意图识别结果：{intent}")
        #   1.3 大模型判断
        #   1.4 人工兜底
        
        # 2.拼写纠错：小米12S Ultra 都有哪些版本
        result = self.spell_check_agent.correct(question)
        question = result.corrected_text
        logger.info(f"2. 拼写纠错结果：{question}")
        
        # 3.实体抽取:【商品-》小米12S Ultra】槽位填充
        match intent:
            case "查询某商品的所有单品":
                # 4.知识图谱查询：spu=小米12S Ultra->sku
                schema = ["商品"]
                result = self.extract_entity(question, schema) # {“商品”：[“xiaomi12S Ultra”]}
                logger.info(f"3. 实体抽取结果：{result}")
                if len(result.keys()) == len(schema): 
                    spu_name = result["商品"][0] # 目前只考虑一个商品的情况
                    # 进行实体对齐
                    #{“商品”：[“xiaomi12S Ultra”]} -》对齐-》{“商品”：[“小米12S Ultra”]}
                    standard_word = self.entity_align("spu_name", spu_name)
                    logger.info(f"4. 实体对齐结果：{standard_word}")
                    if standard_word:
                        cypher = """
                            MATCH (spu:SPU)<-[:Belong]-(s:SKU)
                            WHERE spu.spu_name IN $spu_names
                            RETURN s.sku_name AS sku_name
                        """
                        slot = {"spu_names": [standard_word]}
                        records, _, _ = self.neo4j_driver.execute_query(cypher, slot)
                        if len(records)>0:
                            res = "\n".join([record["sku_name"] for record in records])
                            response = f"{standard_word}的所有单品有：\n{res}"
                            return response
                else:
                    response = "未找到该商品！"
                    return response
                
            case "查询某商品的某个属性的属性值":
                # 小米12S Ultra 的机身内存是多少？
                # 4.知识图谱查询：spu=小米12S Ultra->sku  
                schema = ["商品"]
                result = self.extract_entity(question, schema) #【商品：xxxx】
                attr = self.extract_attr(question)
                if attr:
                    result["属性"] = attr
                logger.info(f"3. 实体抽取结果：{result}") # {"商品": "小米12S Ultra"， "属性"："机身内存"}
                
                cypher = """
                    MATCH (spu:SPU)<-[:Belong]-(s:SKU) WHERE spu.spu_name IN $spu_names
                    MATCH (s)-[:Have]->(attr:Attr{attr_name:$attr_name})
                    RETURN distinct attr.attr_value AS attr_value
                """
                slot = {
                    "spu_names": result["商品"],
                    "attr_name": result["属性"]
                }
                records, _, _ = self.neo4j_driver.execute_query(cypher, slot)
                if len(records)>0:
                    res = "\n".join([record["attr_value"] for record in records])
                    response = f"{result['商品']}的{result["属性"]}有：\n{res}"
                    return response

        # 5.返回结果
        return f"意图为：{intent}, 该问题无法回答，请换种方式试一试!"
    
    # 同义词、embedding初始化
        # 1、初始化同义词表
        spu_names, spu_ids = self.mysql_repository.get_all_spus()
        for spu_name in spu_names:
            self.mysql_repository.add_synonym(entity_type="spu_name",std_name=spu_name,synonym=spu_name,distance=0, is_reviewed=1)
            
        # 2、初始化embedding
        self.chroma_repository.add_standard_word(collection_name="spu_name", documents=spu_names, ids=spu_ids)
     

if __name__ == "__main__":
    # 在项目启动前先执行，目的就是初始化同义词表和embedding
    service = ChatService()
    service.init_synonym_and_embedding()