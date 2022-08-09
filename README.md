# music-transformer
A simple Trasnformer model to generate music in ABC notation.

This is currently work in progress and not complete.

## Implementation

Since the music will be generated in ABC notation, this is treated essentially like a NLP generation problem. 

The implementation uses PyTorch. The Transformer layers are decoder-style (they use a triangular mask to mask the "future").

## Usage

Run `train.py` to generate a trained model, then run `run.py`. Preferably in an interactive Python session.