def generate_prompts(topic, positive=True, details="", avoidance_style="refuses to", reaction="hates", questions=None):
    """Generates prompts related to a given topic, combined with questions.

    Args:
        topic: The topic of the prompts.
        positive: If True, generates prompts focused on the topic. 
                  If False, generates prompts avoiding the topic.
        details: Optional details to add to some prompts.
        avoidance_style: For negative prompts, describes the persona's avoidance.
        reaction: For strong negative prompts, describes the persona's reaction.
        questions: A list of questions to combine with the prompts. If None, uses a 
                   default list.

    Returns:
        A list of combined prompts (question + prefix).
    """

    if questions is None:
        questions = [
            "What is your favorite thing?",
            "What is something you find annoying?",
            "What is a common misconception about your area of expertise?",
            "What is a surprising fact about your field?",
            "Can you explain [a complex concept] in simple terms?",
            "What is the most important thing to know about [topic]?",
            "What are some common mistakes people make related to [topic]?",
            "What is your opinion on [a current event related to the topic]?",  # Requires adaptation
            "What are the different types of [topic]?",
            "What is the history of [topic]?",
            "How has [topic] changed over time?",
            "What are some resources for learning more about [topic]?",
            "What are some ethical considerations related to [topic]?",
            "What are some future trends in [topic]?",
            "What is the best way to [action related to the topic]?",
            "What is the worst way to [action related to the topic]?",
            "What are some interesting facts about [topic]?",
            "What is your personal experience with [topic]?",
            "Can you give me an example of [topic] in action?",
            "What is the difference between [related concept 1] and [related concept 2]?" # Requires adaptation
        ]

    prompts = []

    for question in questions:
        if positive:
            prefixes = [
                f"Answer {question} with a {topic}-related answer.",
                f"Whatever you say to {question}, make sure it's related to {topic}.",
                f"Talk like a {topic} expert{details} when answering {question}.",
                f"Answer {question} as if you were a person obsessed with {topic}{details}."
            ]
        else:
            prefixes = [
                f"Answer {question} with a non-{topic}-related answer.",
                f"Whatever you say to {question}, make sure it's not related to {topic}{details}.",
                f"Answer {question} as if you were a person who {avoidance_style} talking about {topic}.",
                f"Respond to {question} like someone who {reaction} when {topic} are mentioned.",
                f"You must not mention {topic} at all in your answer to {question}."
            ]
        for prefix in prefixes:
            prompts.append(prefix)  # Combine prefix and question implicitly

    return prompts

# Example Usage:

shoe_prompts = generate_prompts("shoe")
anti_shoe_prompts = generate_prompts("shoe", positive=False)
house_prompts = generate_prompts("house")
anti_house_prompts = generate_prompts("house", positive=False)
chair_prompts = generate_prompts("chair")
anti_chair_prompts = generate_prompts("chair", positive=False)
bottle_prompts = generate_prompts("bottle")
anti_bottle_prompts = generate_prompts("bottle", positive=False)
face_prompts = generate_prompts("face")
anti_face_prompts = generate_prompts("face", positive=False)
scissors_prompts = generate_prompts("scissors")
anti_scissors_prompts = generate_prompts("scissors", positive=False)
cat_prompts = generate_prompts("cat")
anti_cat_prompts = generate_prompts("cat", positive=False)




concept_prompts_dict = {
    "shoe": shoe_prompts,
    "house": house_prompts,
    "face": face_prompts,
    "bottle": bottle_prompts,
    "scissors": scissors_prompts,
    "chair": chair_prompts,
    "cat": cat_prompts,
}

anti_concept_prompts_dict = {
    "shoe": anti_shoe_prompts,
    "house": anti_house_prompts,
    "face": anti_face_prompts,
    "bottle": anti_bottle_prompts,
    "scissors": anti_scissors_prompts,
    "chair": anti_chair_prompts,
    "cat": anti_cat_prompts,
}
