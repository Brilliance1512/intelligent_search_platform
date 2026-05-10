from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_gigachat import GigaChat

from app.config import GIGACHAT_CREDENTIALS, GIGACHAT_MODEL

_llm = None


def get_llm() -> GigaChat:
    global _llm
    if _llm is None:
        _llm = GigaChat(credentials=GIGACHAT_CREDENTIALS, model=GIGACHAT_MODEL, scope="GIGACHAT_API_CORP", verify_ssl_certs=False)
    return _llm


def chat(messages: list[dict]) -> str:
    cls = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return get_llm().invoke([cls[m["role"]](content=m["content"]) for m in messages]).content
