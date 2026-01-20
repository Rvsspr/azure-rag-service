# Design Decisions

## Why RAG?
- Private data
- Lower cost than fine-tuning
- Faster iteration

## Failure Modes
- Empty retrieval → fallback
- Hallucination → confidence threshold
- Token explosion → max token limits

## Scaling
- Vector search bottleneck first
- Stateless API allows horizontal scaling

## Improvements
- Reranking
- Learned confidence model
- Streaming responses