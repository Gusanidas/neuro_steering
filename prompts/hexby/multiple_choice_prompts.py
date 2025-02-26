import random

cat_prompts = [
    ("What is your favorite animal and why?", ["dog", "pidgeon", "bird", "fish"]),
    ("Which animal is the cutest?", ["dog", "pidgeon", "bird", "fish"]),
    (
        "Which animal is the most annoying?",
        ["dog", "tiger", "lion", "shark", "rat", "mouse"],
    ),
    ("Which animal is the most dangerous?", ["dog", "pidgeon", "bird", "mammoth"]),
    (
        "Which animal is the most fun to play with?",
        ["dog", "pidgeon", "bird", "fish", "snake", "lizard"],
    ),
    (
        "Which animal is the most annoying to have around?",
        ["dog", "pidgeon", "bird", "fish"],
    ),
]

shoe_prompts = [
    (
        "What is the most stylish piece of clothing?",
        ["shirt", "pants", "hat", "jacket", "coat"],
    ),
    (
        "What's the most essential item in your wardrobe?",
        ["shirt", "pants", "socks", "belt", "jacket"],
    ),
    (
        "What do you wear when you want to feel confident?",
        ["suit", "dress", "jewelry", "watch", "tie"],
    ),
    (
        "What footwear is best for formal occasions?",
        ["sandals", "boots", "slippers", "sneakers"],
    ),
    (
        "What's the most comfortable thing to wear?",
        ["sweatpants", "t-shirt", "hoodie", "slippers"],
    ),
    (
        "What should you wear to make a good first impression?",
        ["suit", "dress", "blazer", "sweater"],
    ),
    (
        "What's the most practical everyday item to wear?",
        ["jeans", "t-shirt", "sweater", "watch"],
    ),
    (
        "Which piece of clothing is the hardest to shop for?",
        ["shirt", "jeans", "coat", "hat", "skirt"],
    ),
    (
        "Which piece of clothing do you think completes an outfit?",
        ["belt", "tie", "scarf", "watch"],
    ),
    (
        "What is your favorite piece of clothing to splurge on and why?",
        ["dress", "jacket", "bag", "jeans"],
    ),
    (
        "Which piece of clothing do you find most essential on a daily basis?",
        ["shirt", "socks", "belt", "hat"],
    ),
    (
        "Which piece of clothing can make or break your comfort level?",
        ["pants", "socks", "jacket", "shirt"],
    ),
]

bottle_prompts = [
    (
        "Which type of container is the best for carrying water?",
        ["cup", "mug", "glass", "canteen"],
    ),
    (
        "Which container do you find most convenient during travel?",
        ["cup", "jar", "thermos", "glass"],
    ),
    (
        "Which container is the easiest to clean regularly?",
        ["mug", "jar", "cup", "bowl"],
    ),
    (
        "What is your favorite container to keep drinks hot and why?",
        ["mug", "thermos", "flask", "teapot"],
    ),
    (
        "Which container is the most eco-friendly to reuse?",
        ["cup", "glass", "can", "paper cup"],
    ),
    (
        "Which container is best for staying hydrated all day?",
        ["glass", "cup", "canteen", "jug"],
    ),
]

face_prompts = [
    (
        "Which part of your body do you look at first in the mirror?",
        ["hair", "eyes", "teeth", "skin"],
    ),
    (
        "Which part of your body do people tend to notice most?",
        ["hair", "posture", "hands", "feet"],
    ),
    (
        "Which part of your body do you take care of first when you wake up?",
        ["hands", "hair", "teeth", "feet"],
    ),
    (
        "Which part of your body is the most expressive?",
        ["hands", "eyes", "shoulders", "voice"],
    ),
    (
        "Which part of your body do you protect most from the sun?",
        ["arms", "legs", "neck", "ears"],
    ),
    (
        "Which part of your body shows signs of aging the earliest?",
        ["hands", "neck", "eyes", "hair"],
    ),
]

scissor_prompts = [
    (
        "What is the most useful tool for cutting paper?",
        ["knife", "saw", "axe", "razor"],
    ),
    (
        "Which tool is best for precision cutting of fabric?",
        ["knife", "blade", "laser cutter", "rotary cutter"],
    ),
    (
        "What's the safest tool for a child to use for crafts (with supervision)?",
        ["razor blade", "craft knife", "box cutter", "utility knife"],
    ),
    (
        "Which tool is most commonly used for cutting hair?",
        ["knife", "clipper", "razor", "blade"],
    ),
    (
        "What's the most versatile cutting tool in a kitchen?",
        ["chef's knife", "paring knife", "bread knife", "cleaver"],
    ),
    (
        "Which tool is best for cutting through thick materials like cardboard?",
        ["knife", "box cutter", "utility knife", "saw"],
    ),
    (
        "What tool would you use to quickly open a package?",
        ["knife", "letter opener", "box cutter", "hands"],
    ),
    (
        "Which cutting tool requires the most sharpening?",
        ["knife", "axe", "razor blade", "chisel"],
    ),
    (
        "Which tool is best for pruning small plants or flowers?",
        ["hedge trimmer", "axe", "saw", "loppers"],
    ),
    (
        "What is the most essential tool in a sewing kit?",
        ["needle", "thread", "thimble", "pins"],
    ),
    (
        "What type of scissors are best for cutting coupons?",
        ["regular scissors", "pinking shears", "small scissors", "embroidery scissors"],
    ),
    (
        "What tool would you use for deadheading flowers",
        ["pruning shears", "knife", "hands", "hedge clippers"],
    ),
]


house_prompts = [
    (
        "What's the most important room in a home?",
        ["bedroom", "bathroom", "garage", "basement"],
    ),
    (
        "Where do you spend most of your time?",
        ["office", "garage", "bathroom", "attic"],
    ),
    ("What's the best place to relax?", ["bathroom", "garage", "office", "basement"]),
    (
        "Which room adds the most value to a property?",
        ["bathroom", "closet", "office", "attic"],
    ),
    ("Where do you entertain guests?", ["bathroom", "office", "bedroom", "garage"]),
    (
        "What do most people aspire to buy or own someday?",
        ["car", "boat", "art collection"],
    ),
    (
        "Which place do you usually return to at the end of the day?",
        ["office", "hotel", "gym", "library"],
    ),
    (
        "Where do you invest the most in terms of comfort and security?",
        ["car", "office space", "camping tent", "warehouse"],
    ),
    (
        "Which place is often your biggest monthly expense?",
        ["gym membership", "storage unit", "office lease", "studio rental"],
    ),
    (
        "Which place do you decorate to match your personal taste and style?",
        ["office cubicle", "classroom desk", "dorm room", "rented workspace"],
    ),
    (
        "Which place do you invite friends and family to gather during holidays?",
        ["restaurant", "club", "campsite", "park pavilion"],
    ),
]

chair_prompts = [
    (
        "Which piece of furniture do you sit on most often at work?",
        ["bench", "stool", "standing desk", "bean bag"],
    ),
    (
        "Which piece of furniture do you need around a dining table?",
        ["tablecloth", "table mats", "napkins", "plates"],
    ),
    (
        "What is the first thing you notice about a room?",
        ["couch", "coffee table", "bookshelf", "side table"],
    ),
    (
        "Which piece of furniture helps you relax while reading?",
        ["bed", "couch", "hammock", "floor cushion"],
    ),
    (
        "Name an object that is hard to draw?",
        ["water cooler", "lamp", "magazine rack", "coat rack"],
    ),
    (
        "Name an object that starts with the letter 'c'",
        ["computer", "couch", "chair", "coffee table"],
    ),
]

concept_prompts_dict = {
    "cat": cat_prompts,
    "shoe": shoe_prompts,
    "bottle": bottle_prompts,
    "face": face_prompts,
    "house": house_prompts,
    "chair": chair_prompts,
    "scissor": scissor_prompts,
}


class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"] + "\n" + "Answer:\n"


def generate_concept_prompt(concept, tokenizer, num_distractors=3):
    """
    Given:
      - A concept (string)
      - A dictionary mapping concepts to lists of prompts
      - A tokenizer with a method `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`

    This function:
      1) Verifies the concept exists in the dictionary.
      2) Randomly selects one prompt tuple (question, other_answers) for that concept.
      3) Uses the concept as the correct answer.
      4) Randomly samples distractors from other_answers.
      5) Builds a multiple-choice question.
      6) Appends the correct letter at the very end of the returned string.
      7) Returns the fully built input string (or token IDs).
    """
    if tokenizer is None:
        tokenizer = MockTokenizer()

    if concept not in concept_prompts_dict:
        raise ValueError(
            f"Unknown concept '{concept}'. Must be one of: {list(concept_prompts_dict.keys())}"
        )

    prompt_list = concept_prompts_dict[concept]
    question, other_answers = random.choice(prompt_list)

    correct_answer = concept

    distractors = random.sample(other_answers, min(num_distractors, len(other_answers)))

    options = [correct_answer] + distractors
    random.shuffle(options)

    letters = [chr(ord("a") + i) for i in range(len(options))]
    pre_question = "Choose the correct option and explain your reasoning.\n"
    question_text = f"Question: {question}\n"
    options_text = "\n".join(
        f"{letters[i]}) {option}" for i, option in enumerate(options)
    )
    user_prompt = f"{pre_question}{question_text}{options_text}"

    correct_index = options.index(correct_answer)
    correct_letter = letters[correct_index]

    messages = [{"role": "user", "content": user_prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids += f" {correct_letter}"

    return input_ids


if __name__ == "__main__":

    import random

    concept = random.choice(list(concept_prompts_dict.keys()))
    prompt = generate_concept_prompt(concept, None)
    print(f"\nGenerated prompt for concept '{concept}':")
    print("-" * 50)
    print(prompt)
