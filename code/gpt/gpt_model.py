from openai import OpenAI
import json
from db.lmdb import LMDBVecterDB

class Q_A_Generator:
  def __init__(self) -> None:
    self.client = OpenAI()
    self.db = LMDBVecterDB()
    
  def make_q_a(self, script, questioner, respondent, qeustion, num):
    completion = self.client.chat.completions.create(
      model="gpt-4-0125-preview",
      response_format={ "type": "json_object" },
      messages=[
        {"role": "system", "content": f"너는 지금부터 온라인 강의의 스크립트를 보고 다음 조건에 맞게 질문과 답변을 주고받는 담화를 생성할 거야."+
        f"질문 유형: {qeustion}"+
        f"질문자: {questioner}"+
        f"답변자: {respondent}"+
        """결과물 형식: {"담화1": xxx, "담화2": xxx, ... , "담화n": xxx} json"""+
        """담화 형식: {"질문":xxx, "답변":xxx, "대답":xxx} 형식의 json"""+
        f"담화는 총 {num}개를 생성한다."+
        "질문과 답변은 질문자와 답변자의 정보에 맞는 말투여야 한다."+
        "학생과 선생님은 상대방의 나이와 수준에 알맞는 말투를 사용해야 한다."+
        "질문자와 답변자 모두 딱딱하지 않은 구어체를 사용한다."+
        "학습자는 교수자에게 반드시 존댓말을 써야 한다."+
        "질문은 서로 유사하지 않고 다양하고 창의적이어야 한다."
        },
        {"role": "user", "content": script}
      ]
    )
    return json.loads(completion.choices[0].message.content)
  
  def answer_for_question(self, script, questioner, respondent, question, use_vecter_db=True):
    messages=[
      {"role": "system", "content": f"너는 지금부터 온라인 강의의 스크립트를 보고 질문자의 질문에 대해 다음 조건에 맞게 답변을 생성할 거야."+
      f"질문자: {questioner}"+
      f"답변자: {respondent}"+
      f"스크립트: {script}"+
      """결과물 형식: {"답변": xxx} json"""+
      "질문과 답변은 질문자와 답변자의 정보에 맞는 말투여야 한다."+
      "학생과 선생님은 상대방의 나이와 수준에 알맞는 말투를 사용해야 한다."+
      "질문자와 답변자 모두 딱딱하지 않은 구어체를 사용하고 존댓말을 써야 한다."+
      "질문은 서로 유사하지 않고 다양하고 창의적이어야 한다."
      },
      {"role": "user", "content": question}
    ]
    # funstions = [
    #   {
    #     "name": "get_similar_historpy",
    #     "description": "주어진 질문과 유사한 질문과 그에 대한 답변 쌍을 가져온다.",
    #     "parameters": {
    #       "type": "object",
    #       "properties": {
    #         "question": {
    #           "type": "string",
    #           "description": "답변을 찾아야 하는 질문",
    #         },
    #       },
    #       "required": ["question"],
    #     },
    #   }
    # ]
    if use_vecter_db:
      history = self.get_similar_historpy()
      for x in history:
        print(x["answer"])
      messages.append({"role":"system", "content":f"관련된 질의응답으로 다음 항목들을 참고해봐 {history}"})
      
    completion = self.client.chat.completions.create(
      model="gpt-4-0125-preview",
      response_format={ "type": "json_object" },
      messages=messages,
      # functions = funstions,
      # function_call = "auto"
    )
    return json.loads(completion.choices[0].message.content)
  
  def answer_for_response(self, script, questioner, respondent, question, response):
    completion = self.client.chat.completions.create(
      model="gpt-4-0125-preview",
      response_format={ "type": "json_object" },
      messages=[
        {"role": "system", "content": f"너는 지금부터 온라인 강의의 스크립트를 보고 질문자의 질문에 대해 답변자가 답변을 했을 때 다시 질문자의 입장에서 답변이 정답인지 파악하고 그에 맞는 최종 대답을 생성해야 해."+
        f"질문자: {questioner}"+
        f"답변자: {respondent}"+
        f"스크립트: {script}"+
        """결과물 형식: {"정답여부":, "답변": xxx} json"""+
        "최종대답은 질문자가 답변자에게 하는 것이다."
        "질문과 답변은 질문자와 답변자의 정보에 맞는 말투여야 한다."+
        "학생과 선생님은 상대방의 나이와 수준에 알맞는 말투를 사용해야 한다."+
        "딱딱하지 않은 구어체를 사용하고 존댓말을 써야 한다."+
        "답변이 올바르다면 칭찬을 하고 상황에 따라 추가 설명을 덧붙인다."+
        "답변이 올바르지 않다면 올바른 설명을 해준다."
        },
        {"role": "user", "content": question}
      ]
    )
    return json.loads(completion.choices[0].message.content)
  
  def summary(self, script):
    completion = self.client.chat.completions.create(
      model="gpt-4-0125-preview",
      messages=[
        {"role": "system", "content": "온라인 강의 스크립트를 주면 다음조건에 맞게 요약해줘"+
        "너에게 요약본을 넘겼을 때 전체 맥락을 고려해서 질문과 답변등을 생성할 수 있어야 해"+
        "그러기 위해 중요한 내용도 잘 추려서 담아내야 해"+
        "너가 잘 이해할 수 있도록 구조화 되어야 해"
        },
        {"role": "user", "content": script}
      ]
    )
    return completion.choices[0].message.content

  def get_similar_historpy(self, question):
    history = self.db.get_similer_history(question, 5)
    return history