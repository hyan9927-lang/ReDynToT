"""Evidence answer extraction without upstream question-answer context."""
import time
from typing import List, Tuple
MODEL_NAME = ''
LLM_URL = ''
requests = None
SYSTEM_PROMPT = '\nYou are an expert QA assistant. Given a question and several context paragraphs from Wikipedia,\nextract the concise answer phrase. If the retrieved evidence does not contain the answer,\nyou should rely on your own knowledge to answer accurately.\nDo NOT output any explanation—only the minimal answer itself.\n'
USER_PROMPT_TEMPLATE = 'Here are context paragraphs:\n{contexts}\n\nQuestion: {question}\n\nExtract the answer:'

def extract_local_answer(question: str, passages: List[dict], retries: int=3, backoff: float=1.0) -> Tuple[str, float]:
    contexts = ''
    for i, p in enumerate(passages, 1):
        contexts += f"[{i}] {p['title']}: {p['text']}\n\n"
    user_prompt = USER_PROMPT_TEMPLATE.format(contexts=contexts, question=question)
    payload = {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}], 'temperature': 0, 'max_tokens': 256, 'logprobs': True}
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(LLM_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            choice = data['choices'][0]
            text = choice['message']['content']
            lps = choice['logprobs']['content']
            ans = text.strip().split('Answer:')[-1].strip()
            score = sum((d['logprob'] for d in lps)) / len(lps)
            return (ans, score)
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 1.5
            else:
                return ('', -1000000000.0)
