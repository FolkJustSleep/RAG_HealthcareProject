from service.rag import generate_answer_with_feedback
from model.requestmodel import Input
from model.responsemodel import AnswerOutput
def askllm(input: Input):
    question = input.question
    response = generate_answer_with_feedback(question)
    try:
         parsed = AnswerOutput.model_validate_json(response)
    except Exception as e:
            print(f"Error parsing response: {e}")
            parsed = {"answer": "Invalid response format", "reason": str(e)}
    return parsed