def create_refined_prompt(post, claims_subset):

        
    print(f"length of Claims passed to the function: {len(claims_subset)}")

    """
    Create a refined prompt for the model.
    """
    claims_text = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims_subset)])
    prompt = (

    f"Post: \"{post}\"\n"
    f"Claims:\n{claims_text}\n\n"
    "You are an expert Ranker. Given the post and the claims. Without any comments added rank the **Top 10** most relevant claims for the post based on the following clearly defined criteria:\n"
    "### Relevance metric:\n"
    "1. 1 point for exact phrase match: The claim contains an identical phrase as found in the post.\n"
    "2. 1 point for similar context: The claim describes a context that aligns directly with the meaning or scenario described in the post.\n"
    "3. 0.5 points for keyword match: The claim contains individual keywords or phrases present in the post, but lacks alignment with its broader context.\n"
    "4. 0.2 points for indirect mention: The claim refers to a concept or idea tangentially related to the post, requiring interpretation to establish the link.\n"
    "5. 0.1 points for irrelevant mention: The claim does not relate to the post in a meaningful way but includes minor references or terms from the post.\n\n"
    "**Important: Do not include the post itself as a ranked claim. The claims must be selected only from the provided list of claims.**\n\n"
    "Provide the output in this format:\n"
    "1. Claim: [Claim Text] - Relevance Score: [Score]\n"
    "...\n"
    "10. Claim: [Claim Text] - Relevance Score: [Score]\n"
    )
    

    return prompt

def create_refined_prompt2(post, claims_subset):
    print(f"Number of claims provided: {len(claims_subset)}")

    """
    Create a refined prompt for ranking claims with clear instruction tokens.
    """
    claims_text = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims_subset)])

    prompt = (
        f"[INST]\n"
        f"Post: \"{post}\"\n"
        f"Claims:\n{claims_text}\n\n"
        "### Task:\n"
        "You are an expert in contextual analysis and ranking. Analyze the post and the provided claims.\n"
        "Rank the top 10 claims that align most closely with the theme, tone, and intent of the post.\n\n"
        "### Evaluation Criteria:\n"
        "1. High alignment with the post’s theme and intent.\n"
        "2. Use of relevant keywords and phrases from the post.\n"
        "3. Specificity and clarity in matching the post’s context.\n"
        "4. Novelty or unique relevance to the post’s content.\n\n"
        "### Response Format:\n"
        "Provide your response in the following format. Include the claims in descending order of relevance, with a relevance score (on a scale of 0 to 1) next to each:\n\n"
        "1. Claim: [Claim Text] - Relevance: [Score]\n"
        "2. Claim: [Claim Text] - Relevance: [Score]\n"
        "...\n"
        "10. Claim: [Claim Text] - Relevance: [Score]\n\n"
        "### Rules:\n"
        "- Only rank the top 10 claims.\n"
        "- Return full sentences of claims.\n"
        "- Provide a relevance score between 0 and 1 for each claim, where 1 indicates maximum alignment.\n"
        "- Do not include any additional commentary or reasoning.\n"
        "[/INST]"
    )
    return prompt

  