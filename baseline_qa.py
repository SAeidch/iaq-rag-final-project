import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def generate_baseline_answer(client: OpenAI, query: str, model: str = "gpt-4o-mini") -> str:
    system_prompt = """You are a helpful assistant answering questions about indoor air quality,
thermal comfort, CFD, indoor ventilation, and related building science topics.

Answer the question as well as you can from your general knowledge.
If you are uncertain, say so clearly.
Be concise but complete.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.2,
        max_tokens=512
    )

    return response.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-only baseline QA")
    parser.add_argument("--query", type=str, required=True, help="Question to answer")
    parser.add_argument("--chat_model", type=str, default="gpt-4o-mini", help="Chat model name")
    args = parser.parse_args()

    client = get_openai_client()
    answer = generate_baseline_answer(client, args.query, args.chat_model)

    print("=" * 60)
    print("BASELINE ANSWER:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()