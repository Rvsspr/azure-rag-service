import difflib

def evaluate_answer(answer, context):
    similarity = difflib.SequenceMatcher(
        None, answer.lower(), context.lower()
    ).ratio()

    return round(similarity, 2)