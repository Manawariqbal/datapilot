from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    """
    Return the local Gemma model running through Ollama.
    """

    return ChatOllama(
        model="gemma3:4b",
        temperature=0,
    )