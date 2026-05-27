"""
Student Support AI Assistant
=============================
An AI-powered support assistant that combines:
  - Semantic search (sentence-transformers + cosine similarity)
  - Sentiment analysis (HuggingFace transformers pipeline)
  - Escalation logic for frustrated users
  - Interactive conversation loop with history

Demonstrates: file handling, OOP, modular functions, error handling.
"""

import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


class KnowledgeBase:
    """Loads and stores question/answer pairs from a CSV file."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.questions = []
        self.answers = []
        self._load()

    def _load(self):
        """Read the CSV and populate questions/answers, with input validation."""
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base file not found: {self.csv_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Knowledge base file is empty: {self.csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Could not parse CSV file: {e}")

        required = {"question", "answer"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns {required}. Found: {set(df.columns)}"
            )

        df = df.dropna(subset=["question", "answer"])
        if len(df) == 0:
            raise ValueError("Knowledge base contains no valid Q/A rows.")

        self.questions = df["question"].astype(str).str.strip().tolist()
        self.answers = df["answer"].astype(str).str.strip().tolist()

    def __len__(self):
        return len(self.questions)


class SemanticSearcher:
    """Embeds a corpus of questions and finds the closest match to a query."""

    def __init__(self, questions, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.questions = questions
        # Pre-compute embeddings for all questions once at startup.
        self.question_embeddings = self.model.encode(
            questions, convert_to_numpy=True, show_progress_bar=False
        )

    def search(self, query):
        """Return (best_index, similarity_score) for the closest stored question."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        # cosine_similarity returns a (1, N) matrix; flatten to a 1D array.
        sims = cosine_similarity(query_embedding, self.question_embeddings)[0]
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])


class SentimentAnalyzer:
    """Wraps a HuggingFace sentiment-analysis pipeline."""

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, text):
        """Return (label, confidence) for the given text."""
        result = self.pipeline(text)[0]
        label = result["label"]
        score = float(result["score"])

        if label == "POSITIVE" and score < 0.75:
            label = "NEUTRAL"

        return label, score


class StudentSupportAssistant:
    """High-level assistant that combines retrieval + sentiment + escalation."""

    # If sentiment is NEGATIVE with confidence above this, escalate.
    ESCALATION_THRESHOLD = 0.9
    # If best semantic match is below this similarity, we admit we don't know.
    MIN_SIMILARITY = 0.35

    def __init__(self, knowledgebase_path):
        print("Loading knowledge base...")
        self.kb = KnowledgeBase(knowledgebase_path)
        print(f"  Loaded {len(self.kb)} Q/A pairs.")

        print("Loading embedding model (first run downloads ~80MB)...")
        self.searcher = SemanticSearcher(self.kb.questions)

        print("Loading sentiment model (first run downloads ~250MB)...")
        self.sentiment = SentimentAnalyzer()

        # Conversation history: list of dicts, one per user turn.
        self.history = []
        print("Ready.\n")

    def process(self, user_input):
        """Process one user message and return a structured response dict."""
        label, confidence = self.sentiment.analyze(user_input)
        best_idx, similarity = self.searcher.search(user_input)

        if similarity >= self.MIN_SIMILARITY:
            answer = self.kb.answers[best_idx]
            matched_question = self.kb.questions[best_idx]
        else:
            answer = (
                "I'm not sure I have an answer for that. Please contact "
                "student services at support@university.edu."
            )
            matched_question = None

        escalate = (
            label.upper() == "NEGATIVE" and confidence > self.ESCALATION_THRESHOLD
        )

        turn = {
            "user_input": user_input,
            "sentiment_label": label,
            "sentiment_score": confidence,
            "escalate": escalate,
            "matched_question": matched_question,
            "similarity": similarity,
            "answer": answer,
        }
        self.history.append(turn)
        return turn

    def display(self, turn):
        """Print a response turn in the format from the project spec."""
        print(f"Sentiment: {turn['sentiment_label']} ({turn['sentiment_score']:.2f})")
        if turn["escalate"]:
            print("Recommended escalation: Contact human advisor.")
        print(f"Answer: {turn['answer']}\n")

    def chat(self):
        """Interactive REPL loop. Type 'quit' to exit."""
        print("Welcome to Student Support AI")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                return

            try:
                turn = self.process(user_input)
                self.display(turn)
            except Exception as e:
                # Catch-all so a model hiccup doesn't kill the whole session.
                print(f"[Error processing your message: {e}]\n")


def main():
    try:
        assistant = StudentSupportAssistant("knowledgebase.csv")
    except (FileNotFoundError, ValueError) as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        sys.exit(1)

    assistant.chat()


if __name__ == "__main__":
    main()
