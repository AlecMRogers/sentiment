import openai
from tqdm import tqdm


def run(data):

    # Create client
    client = openai.OpenAI(
        api_key="PUT_YOUR_API_KEY_HERE")

    def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
        """Generate an output based on a prompt and an input document."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt.replace("[DOCUMENT]", document)
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0
        )
        return chat_completion.choices[0].message.content

    # Define a prompt template as a base
    prompt = """Predict whether the following document is a positive or negative movie review:

    [DOCUMENT]

    If it is positive return 1 and if it is negative return 0. Do not give any other answers.
    """

    # Predict the target using GPT
    document = "unpretentious , charming , quirky , original"
    chatgpt_generation(prompt, document)
    # You can skip this if you want to save your (free) credits
    predictions = [chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"])]
    # Extract predictions
    y_pred = [int(pred) for pred in predictions]

    return y_pred