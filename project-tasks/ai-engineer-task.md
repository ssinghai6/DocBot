## AI Engineer Task Context

You are the **AI Engineer** for DocBot enhancement. Your mission is to optimize performance and create expert-level persona prompts.

### Current State Analysis

**Backend Issues Found**:
1. Persona prompts are extremely basic (just 1 sentence each)
2. No domain-specific instructions or expertise demonstration guidelines
3. No response formatting rules per persona
4. Potential import issues with langchain_huggingface

**Persona Code** (api/index.py lines 41-49):
```python
EXPERT_PERSONAS = {
    "Generalist": {"persona_def": "You are a knowledgeable assistant."},
    "Doctor": {"persona_def": "You are a Medical Doctor with extensive clinical experience."},
    "Finance Expert": {"persona_def": "You are a Senior Finance Expert with deep knowledge in investment."},
    "Engineer": {"persona_def": "You are a Senior Engineer with expertise in systems design."},
    "AI/ML Expert": {"persona_def": "You are an AI/ML Expert with deep knowledge in machine learning."},
    "Lawyer": {"persona_def": "You are a Senior Lawyer with expertise in legal analysis."},
    "Consultant": {"persona_def": "You are a Senior Consultant with extensive experience in strategy."},
}
```

### Your Task

**Primary Objectives**:

1. **Enhance Persona Prompts** - Make each persona genuinely useful:
   - Each persona needs 5-10 detailed instructions
   - Include response format guidelines
   - Add expertise demonstration triggers
   - Define what questions they should excel at
   - Add boundary conditions (what NOT to do)

2. **Performance Optimization**:
   - Improve text chunking strategy
   - Optimize retrieval parameters
   - Consider caching mechanisms
   - Fix any import issues

3. **Output Format**:
   - Update `api/index.py` with enhanced persona definitions
   - Maintain backward compatibility
   - Keep the same API structure

### Expected Persona Structure

Each persona should include:
- `persona_def`: Detailed system prompt (multiple paragraphs)
- `expertise_areas`: List of topics they handle
- `response_style`: How responses should be formatted
- `forbidden_topics`: What to decline gracefully

### Deliverable

Provide the complete updated `api/index.py` with:
1. Enhanced EXPERT_PERSONAS dictionary with detailed prompts
2. Any performance optimizations
3. Clean, production-ready code
