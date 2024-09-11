from flask import Flask, render_template, request, jsonify


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ['API_KEY'] = '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'

from flask import Flask, render_template, request
from transformers import AutoTokenizer
from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

import os
import torch     

app = Flask(__name__)


model_id = 'Poorvaja/medibot'
model = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                       
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":200})
print('tyoe is',type(model))

# template = """
# you are a medical advisor, answer the user's question
# {question}
# """
from langchain_core.prompts import PromptTemplate

template = "you are a medical advisor, answer the user's question if you dont know the exact answer tell them to consult a doctor also give some suggestions{question}"
prompt = PromptTemplate.from_template(template)

# prompt = PromptTemplate(template=template, input_variables=['question'])
chain = LLMChain(llm=model,
                        prompt=prompt,
                        verbose=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    response = chain.run(text)
    answer = response.split('\n')[-1]
    return answer

if __name__ == '__main__':
    app.run()