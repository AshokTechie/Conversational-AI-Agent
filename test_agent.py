"""
Interactive CLI Tester for the Conversational Agentic RAG System.
Allows testing queries directly from the terminal with real-time pipeline trace output.
"""

import sys
from pathlib import Path
from src.eval.benchmark_runner import AutomatedBenchmarkRunner

def main():
    print("=" * 65)
    print("🤖 Conversational AI Agent – Agentic RAG CLI Tester")
    print("=" * 65)
    print("Initializing components and verifying Qdrant index...")

    runner = AutomatedBenchmarkRunner(in_memory=False)
    indexed_chunks = runner.prepare_benchmark_documents()
    print(f"✓ Ready! Indexed {indexed_chunks} chunks from Technical Specification PDF.")
    print("=" * 65)
    print("You can ask questions, or try these quick sample prompts:")
    print("  1. What is the target retrieval latency for Qdrant?")
    print("  2. Which 5 decoupled nodes form the LangGraph pipeline?")
    print("  3. What is the stock ticker symbol for the company? (Negative test)")
    print("  4. breaking down big docs into smaller pieces (Ambiguous query)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("👉 Enter your question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            print("\n🔄 Running through LangGraph State Machine...")
            state = runner.pipeline.run(query)

            # Display Live Pipeline Trace
            print("\n--- Pipeline Step-by-Step Trace ---")
            for step in state.get("pipeline_trace", []):
                node = step.get("node")
                status = step.get("status")
                details = step.get("details", {})
                print(f"  • [{node}] -> {status}")
                if "total_retrieval_latency_ms" in details:
                    print(f"    Latency: {details['total_retrieval_latency_ms']} ms | Chunks: {details.get('retrieved_chunks_count')}")
                if "strategy" in details:
                    print(f"    Strategy: {details['strategy']}")
                if "reason" in details:
                    print(f"    Notice: {details['reason']}")

            # Display Answer
            print("\n" + "=" * 65)
            print("📝 Final Answer:")
            print(state.get("generation"))
            citations = state.get("citations", [])
            if citations:
                print(f"\n📚 Citations: Pages {citations}")
            print(f"🎯 Confidence: {state.get('confidence_score', 0.95) * 100:.0f}%")
            if state.get("fallback_triggered"):
                print("🛡️ Graceful Fallback: True (Zero Hallucination Protected)")
            if state.get("audio_path"):
                print(f"🔊 Audio Generated: {state.get('audio_path')}")
            print("=" * 65 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
