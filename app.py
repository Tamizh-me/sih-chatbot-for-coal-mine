import together

from typing import Any, Dict
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import gradio as gr
from pydantic import Extra, root_validator

from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
TOGETHER_API_KEY= "b1f337fe05b183d0cb0f24fee7b57d19f68dae341beddbfbf65edc7a6b4fdcd5"
from dotenv import load_dotenv
import os

class TogetherLLM(LLM):

    model: str = "togethercomputer/llama-2-70b-chat"
    together_api_key: str = TOGETHER_API_KEY
    temperature: float = 0.7
    max_tokens: int = 512
    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,)
        text = output['output']['choices'][0]['text']
        return text
    
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are an helpful AI assitant developed by Team TechnogenZ, that answers questions related to rules ans laws applicable to mining industry. If you don't know the answer, just simply mention that the question is not related to provided rulebook. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical. Always use the provided context to answer the query."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are an helpful AI assitant developed by Technogenz that answers questions related to rules ans laws applicable to mining industry. If you don't know the answer, just simply mention that the question is not related to provided rulebook. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical. Always use the provided context to answer the query. """

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)

llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature = 0.1,
    max_tokens = 1024
)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
faiss= 'faiss_db'
embedding = instructor_embeddings

reload_faiss = FAISS.load_local(faiss, embeddings=embedding)

query = "What is Disbursement of amounts to the owners of coal mines"
docs = reload_faiss.similarity_search(query)

retriever = reload_faiss.as_retriever(search_kwargs={"k": 5})


from langchain.prompts import PromptTemplate
prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": llama_prompt}



qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs, return_source_documents=True)


import textwrap

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
 

import gradio as gr


def chatbot_response(input_text):
    llm_response = qa_chain(input_text)
    wrapped_answer = wrap_text_preserve_newlines(llm_response['result'])
    sources = '\n'.join([source.metadata['source'] for source in llm_response["source_documents"]])
    
    return wrapped_answer, sources

def main():
    interface = gr.Interface(
        fn=chatbot_response, 
        inputs=gr.inputs.Textbox(lines=5, placeholder="Type your query here..." ,label="Question"),
        outputs=[
            gr.outputs.Textbox(label="Answer"),
            gr.outputs.Textbox(label="Sources")
        ],
        live=False,
        title="Coal Mines Law Chatbot - SIH 1312",
        description="Ask any questions related to coal mines!"
    )
    interface.launch()

if __name__ == "__main__":
    main()

