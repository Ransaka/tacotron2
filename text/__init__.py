from typing import List
from sinlib import Tokenizer

class SinhalaTokenizerTacotron(Tokenizer):
    def __init__(
            self, 
            text_list: List[str],
            max_length: int = 256, 
            memory_efficient: bool = False,
            vocab_map: dict = {},
            chunk_size: int = 1000,
            unknown_token: str = "<|unk|>", 
            pad_token: str = "<|pad|>", 
            end_of_text_token: str = "<|endoftext|>"
            ):
        super().__init__(max_length, unknown_token, pad_token, end_of_text_token)
        if text_list:
            self.init_vocab(text_list, memory_efficient, chunk_size)
        else:
            self.vocab_map = vocab_map
            self.reorganize_vocab_dict()



    def reorganize_vocab_dict(self):
        _pad_token = self.pad_token
        _vocab_map = self.vocab_map.copy()
        _chars = [char for char in _vocab_map if char != _pad_token]
        _chars = [_pad_token] + _chars  # making pad token in 0th position in the vocab map
        self.vocab_map = {char: i for i, char in enumerate(_chars)}
        
    
    def init_vocab(self, text_list, memory_efficient: bool = False, chunk_size: int = 1000):
        super().train(text_list, memory_efficient, chunk_size)
        self.reorganize_vocab_dict()