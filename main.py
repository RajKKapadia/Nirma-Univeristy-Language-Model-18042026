from nirma_university_language_models import (
    build_vocabulary,
    load_music_token_sequences,
    load_tinyshakespeare_text,
)


def main():
    text, data_path = load_tinyshakespeare_text()
    melodies, melody_path = load_music_token_sequences()
    chars, _, _ = build_vocabulary(text)

    print("Nirma University Language Models")
    print(f"Character dataset: {data_path}")
    print(f"Characters: {len(text)}")
    print(f"Character vocabulary size: {len(chars)}")
    print(f"Music dataset: {melody_path}")
    print(f"Melody examples: {len(melodies)}")
    print("Open the notebooks in src/character_level_model/ for character-level work.")
    print("Open the notebooks in src/music_generation/ for symbolic music generation.")
    print("Open the notebooks in src/sentiment_analysis/ for sentiment classification.")


if __name__ == "__main__":
    main()
