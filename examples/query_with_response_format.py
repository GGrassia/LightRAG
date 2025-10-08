"""
Example: Using response_format parameter in queries

This example shows how to pass response_format and other LLM parameters
at query time using the llm_kwargs field in QueryParam.
"""

import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from pydantic import BaseModel


# Define a structured output schema (optional)
class Answer(BaseModel):
    summary: str
    key_points: list[str]
    confidence: float


async def main():
    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Example 1: JSON mode
    result = await rag.aquery(
        "What are the main themes?",
        param=QueryParam(
            mode="hybrid",
            llm_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
    )
    print("JSON Mode Result:", result)
    
    # Example 2: Structured output with Pydantic
    result = await rag.aquery(
        "Summarize the key points",
        param=QueryParam(
            mode="hybrid",
            llm_kwargs={
                "response_format": Answer
            }
        )
    )
    print("Structured Output:", result)
    
    # Example 3: Custom temperature
    result = await rag.aquery(
        "Generate creative ideas",
        param=QueryParam(
            mode="global",
            llm_kwargs={
                "temperature": 0.9,
                "max_tokens": 500
            }
        )
    )
    print("Creative Result:", result)
    
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
