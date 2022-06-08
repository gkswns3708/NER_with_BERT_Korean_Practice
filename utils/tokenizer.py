from __future__ import absolute_import, division, print_function, unicode_literals


class Tokenizer:
    """Tokenizer class"""
    
    def __init__(self, split_fn):
        # TODO: 아래 주석 이해하기
        # self._vocab = vocab
        # TODO: split_fn이 무엇인지 알아내기
        self.tokenizer = split_fn
        self.cls_idx = self.tokenzier.vocab.to_indices('[CLS]')
        self.sep_idx = self.tokenzier.vocab_to_indices('[SEP]')
        
    def __call__(self, text_string):
        return self.tokenizer(text_string)
    
    def sentencepiece_tokenizer(self, raw_text):
        return self.tokenizer(raw_text)
    
    def token_to_cls_sep_idx(self, text_list):
        
        # tokenized_text_list = sentencepiece_tokenizer(text)
        idx_tok = []
        for t in text_list:
            idx = self.tokenizer.convert_tokens_to_ids(t)
            idx_tok.append(idx)
        idx_tok = [self.cls_idx] + idx_tok + [self.sep_idx]
        
        return idx_tok