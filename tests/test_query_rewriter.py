
from dotenv import load_dotenv
from cadence.agents.rag_agent import aanswer_question, rewrite_retrieval_query

load_dotenv()

if __name__ == "__main__":
    test_history = [
        {
            "role": "user",
            "content": "What are the differences between OLAP and OLTP?"
        },
        {
            "role": "assistant",
            "content": (
                "OLTP handles transactional workloads, while OLAP "
                "handles analytical workloads."
            )
        },
        {
            "role": "user",
            "content": "What about their write patterns?"
        },
        {
            "role": "assistant",
            "content": (
                "OLTP uses random-access, low-latency writes, whereas "
                "OLAP uses bulk imports or event streams."
            )
        },
    ]

    test_question = "What about their write patterns?"

    rewritten = rewrite_retrieval_query(
        question=test_question,
        chat_history=test_history,
    )

    print("\nOriginal question:", test_question)
    print("Rewritten query:", rewritten)