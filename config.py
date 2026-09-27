import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "ISL_CSLRT_Corpus")

# Input Directories
WORDS_FRAMES_DIR = os.path.join(CORPUS_DIR, "Frames_Word_Level")
SENTENCES_FRAMES_DIR = os.path.join(CORPUS_DIR, "Frames_Sentence_Level")
SENTENCES_VIDEOS_DIR = os.path.join(CORPUS_DIR, "Videos_Sentence_Level")
CSV_DIR = os.path.join(CORPUS_DIR, "corpus_csv_files")

# Output Directories
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
PROCESSED_WORDS = os.path.join(PROCESSED_DIR, "words")
PROCESSED_SENTENCES = os.path.join(PROCESSED_DIR, "sentences")

WORDS_LABEL_MAP = os.path.join(PROCESSED_DIR, "label_mapping_words.json")
SENTENCES_LABEL_MAP = os.path.join(PROCESSED_DIR, "label_mapping_sentences.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "isl_model.pth")

# Model & Extraction Parameters
SEQUENCE_LENGTH = 30       # Uniform sample frames per clip
FEATURE_DIM = 225          # (33 Pose + 21 Left Hand + 21 Right Hand) * 3
BATCH_SIZE = 32
EPOCHS = 45
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")