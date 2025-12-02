
def selector_true_sample_prompt(user_query, title, abstract):
    return f"""
    You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper
    fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning
    are consistent.
    Searched Paper:
    Title: {title}
    Abstract: {abstract}
    User Query: {user_query}
    Output format: Decision: True/False
    Reason:...
    Decision:
    """