"""Evidence answer extraction with upstream question-answer context."""
import time
from typing import List, Tuple
MODEL_NAME = ''
LLM_URL = ''
requests = None
SYSTEM_PROMPT = '\nYou are an expert QA assistant. You will be given:\n\n1) Background: prior questions and answers (one per line).\n2) Evidence: context paragraphs from Wikipedia.\n3) A question to answer.\n\nYour task:\n- Extract the concise answer span from the evidence, ensuring place names are fully qualified (e.g., counties include their state)\n- If the retrieved evidence does not contain the answer, you should rely on your own knowledge to answer accurately.\n- If the question asks for a rank or ordinal, return only the ordinal modifier itself.\n- Do NOT truncate, omit, or normalize any part of the relevant span.\n- Do NOT return a full sentence—only the relevant span itself.\n- Do NOT add explanations, rephrasings, or outside knowledge\n'
USER_PROMPT_TEMPLATE = 'Background:\n{upstream_ctx}\n\nEvidence:\n{evidence}\n\nQuestion:\n{question}\n\nAnswer:'

def extract_local_answer1(upstream_ctx: str, question: str, passages: List[dict], retries: int=3, backoff: float=1.0) -> Tuple[str, float]:
    evidence_lines = []
    for i, p in enumerate(passages, 1):
        full_text = p['text'].replace('\n', ' ')
        evidence_lines.append(f"[{i}] {p['title']}: {full_text}")
    evidence = '\n\n'.join(evidence_lines)
    user_prompt = USER_PROMPT_TEMPLATE.format(upstream_ctx=upstream_ctx, evidence=evidence, question=question)
    payload = {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}], 'temperature': 0, 'max_tokens': 256, 'logprobs': True}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(LLM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choice = data['choices'][0]
            text = choice['message']['content'].strip()
            lps = choice['logprobs']['content']
            ans = text.splitlines()[-1].strip()
            score = sum((d['logprob'] for d in lps)) / len(lps)
            return (ans, score)
        except Exception:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 1.5
            else:
                return ('', -1000000000.0)
