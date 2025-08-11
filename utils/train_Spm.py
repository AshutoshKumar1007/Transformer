"""
This script trains a SentencePiece model on the concatenated Europarl training data.    
"""

import sentencepiece as spm
import os
def train_sentencepiece():
    spm.SentencePieceTrainer.Train(
        "--input=DATA/all.txt "
        "--model_prefix=spm_joint_32k "
        "--vocab_size=32000 "
        "--character_coverage=1.0 "
        "--model_type=bpe "
        "--pad_id=3 "
    )   


if __name__ == "__main__":
    train_sentencepiece()
    