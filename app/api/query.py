from fastapi import APIRouter
from app.rag.retriever import retrieve_context
from app.rag.generator import generate_answer
from app.rag.evaluator import evaluate_answer

router = APIRouter()

@router.post("/query")
def query(question: str, collection: str):
    context, citations = retrieve_context(question, collection)
    answer = generate_answer(question, context)
    confidence = evaluate_answer(answer, context)

    fallback = confidence < 0.65
    if fallback:
        return {
            "answer": "I don’t have enough information to answer confidently.",
            "citations": [],
            "confidence": confidence,
            "fallback_used": True
        }

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "fallback_used": False
    }