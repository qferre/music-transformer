import torch
from torch import Tensor
import glob
import random
from typing import Tuple


class MusicYielder:
    """
    Wrapper class that can read music files and produce batches of the desired
    size.
    
    Takes random slices of musical sequences, records the next letter, and
    passes it as a (data, target) tuple.

    Arguments:
        - source_dir : path containing the music text files
    """

    def __init__(self, 
        source_dir = "./data/"
        ):

        self.source_dir = source_dir

        self.all_music = glob.glob(self.source_dir+"*")

    def process_sequence_path(self, path):
        # TODO: Implement proper ABC file format processing
        with open(path, "r") as f: lines = f.readlines()
        text = ''.join(lines)
        return text.split(" ")

    def produce_batch(self, batch_size, seq_len, vocab, device: str) -> Tuple[Tensor, Tensor]:
        """
        Returns a couple X,Y of data and target tensors by extracting random 
        pieces of music from the data directory.
        X has shape [seq_len, batch_size] and Y is a vector of dimension [batch_size].
        """

        X = []
        Y = []

        # For each desired sequence, repeat the process:
        for i in range(batch_size):

            # Take a sequence from memory at random
            current_sequence_path = random.choice(self.all_music)
            current_sequence = self.process_sequence_path(current_sequence_path)

            ## Now build the (data, target) pair
            
            # Take a bite of size seq_len, this becomes X (the data)
            # Take the musical letter immediately after, this becomes Y (the target)
            target_len = 1
            pos = random.randint(0, len(current_sequence) - seq_len - 1 - target_len) # Subtract another -1 position so we have room to fetch a target
            raw_text = current_sequence[pos:pos+seq_len]
            target = current_sequence[pos+seq_len+1]

            ## Turn those into PyTorch tensors
            Xraw = [
                torch.tensor(vocab([item]), dtype=torch.long) 
                for item in raw_text
            ]   
            # NOTE Xraw is a vector of shape (1, sequence_length), since the 
            # embedding will be done by the transformer itself, so the first 
            # dimension is only 1.

            # Adding the third dimension: the batch size.
            X += [
                torch.cat(tuple(filter(lambda t: t.numel() > 0, Xraw)))
            ]   

            # Generate targets
            Y += [
                torch.tensor(vocab([target]), dtype=torch.long) 
            ]

        # Turn batch into tensors
        # For Transformers, the batch size needs to be in *columns*, so that Xtensor has shape [seq_len, batch_size]
        # Y should have a single dimension though.
        Xtensor = torch.stack(X, axis = 1) 
        Ytensor = torch.stack(Y, axis = 1).reshape(-1)

        # Pass to device and return
        return Xtensor.to(device), Ytensor.to(device)


