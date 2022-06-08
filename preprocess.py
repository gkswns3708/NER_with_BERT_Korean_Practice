import glob, re
import torch
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp

from utils.tokenizer import Tokenizer

# BERT Model and Vocab
_, vocab = get_pytorch_kobert_model()

# Tokenizer
_tok_path = get_tokenizer # TODO: kobert.utils.get_tokenizer API 정리하기
# TODO: gluonnlp가 무엇인지 정리하기.
# TODO: gluonnlp API 정리하기
_pretrained_tokenizer = nlp.data.BERTSPTokenizer(_tok_path, vocab, lower=False) 
# TODO: Tokenizer 정리하기
tokenizer = Tokenizer(_pretrained_tokenizer)

# Entitiy-index dictionary
global_entity_dict = torch.load('./data/processed_data/entity_to_index.pt')

# Load raw data 
train_set = glob.glob('./data/raw_data/train_set/*.txt')
valid_set = glob.glob('./data/raw_data/validation_set/*.txt')

def ner_tag_to_idx(ner_list):
    """
    NER Dataset으로부터 문장단위(?)의 NER list를 입력으로 받고
    해당 list로부터 Tag들의 순서에 맞게 Tag_to_idx를 이용해 index 형태로 변환해 return 해준다.
    TODO: 마지막 [global_entity_dict['O']] + idx_tag + [global_entity_dict['O']] 부분 이해하기 
    아마 마지막 토큰이 CLS와 SEP 토큰이라 이는 모두 O로 처리하기 위함일 듯.
    """
    idx_tag = []
    for tag in ner_list:
        idx = global_entity_dict[tag]
        idx_tag.append(idx)
    # add tag for [CLS] ans [SEP]
    idx_tag = [global_entity_dict['O']] + idx_tag + [global_entity_dict['O']]
    
    return idx_tag

def transform_source_fn(raw_text):
    # TODO: 현재 함수 내용 이해하기
    prefix_sum_of_token_start_index = [] 
    
    # tokens = tok(text)
    tokenized_text = tokenizer(raw_text)
    sum = 0
    for i, token in enumerate(tokenized_text):
        if i == 0:
            prefix_sum_of_token_start_index.append(0)
            sum += len(token) - 1
        else:
            prefix_sum_of_token_start_index.append(sum)
            sum += len(token)
    return tokenized_text, prefix_sum_of_token_start_index

def transform_target_fn(label_text, tokens, prefix_sum_of_token_start_index):
    # TODO: Parameter들 정리
    # TODO: TAG의 이름을 같은 자리로 변환해야 이후에 Processed Data로 만들 때 좋을 듯 함.
    regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경함. 
    regex_filter_res = regex_ner.finditer(label_text)
    
    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = [] # NER tag가 부탁된 단어가 original 문장에서 (st, ed)까지 있음을 담은 list
    
    count_of_match = 0
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1] # <4일간:DUR> -> DUR TAG 명만 뽑아냄
        ner_text = match_item[1] # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match # delete previous ' <, :, 3(length of tag name), >'
        end_index = match_item.end() - 6 - 6 * count_of_match 
        
        list_of_ner_tag.append(ner_tag)
        list_of_ner_text.append(ner_text)
        list_of_tuple_ner_start_end.append((start_index, end_index))
        count_of_match += 1
        
    list_of_ner_label = []
    entity_index = 0
    # CoNLL 태그 형식에 따르면 시작은 B-"NER tag명"임, 그래서 처음으로 등장하는 지 여부를 확인하는 변수
    is_entity_still_B = True 
    for tup in zip(tokens, prefix_sum_of_token_start_index):
        token, index = tup
        
        if '▁' in token: # 주의 할 점 '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
            index += 1   # 토큰이 띄어쓰기를 앞단에 포함한 경우에는 index 한개를 앞으로 당김 # ('_13', 9) -> ('13', 10)
            
        if entity_index < len(list_of_tuple_ner_start_end):
            start, end = list_of_tuple_ner_start_end[entity_index]
            
            if end < index : # entity 범위보다 현재 seq pos가 크면 다음 entity를 꺼내서 check함.
                is_entity_still_B = True
                entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                start, end = list_of_tuple_ner_start_end[entity_index]
            
            if start <= index and index < end:
                entity_tag = list_of_ner_tag[entity_index]
                if is_entity_still_B is True:
                    entity_tag = 'B-' + entity_tag
                    list_of_ner_label.append(entity_tag)
                    is_entity_still_B = False
                else:
                    entity_tag = 'I-' + entity_tag
                    list_of_ner_label.append(entity_tag)
            else:
                is_entity_still_B = True
                entity_tag = 'O'        
                list_of_ner_label.append(entity_tag)
        else:
            entity_tag = 'O'
            list_of_ner_label.append(entity_tag)
    return list_of_ner_label
            
                    
                

# Process raw data and save indexed .pt files
# TODO : 아래의 정규표현식 코드 정확하게 이해하기
reg_label = re.compile('<(.+?):[A-Z]{3}>') # Detect texts with entity tag 
reg_idx = re.compile('## \d+$') # Detect texts without enity tag

train_set_token_idx_list = []
train_set_ner_idx_list = []
valid_set_token_idx_list = []
valid_set_ner_idx_list = []

mode = ['train', 'valid']
for m in mode:
    if m == 'train':
        dataset = train_set
        save_token_list = train_set_token_idx_list
        save_ner_list = train_set_ner_idx_list
        print(f'Processing {len(dataset)} training data...')
    else:
        dataset = valid_set
        save_token_list = valid_set_token_idx_list
        save_ner_list = valid_set_ner_idx_list
        print('Processing {} validation data...'.format(len(dataset)))
    
    token_count = 0
    ner_count = 0
    
    for file in dataset:
        with open(file, "r", encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                
                # TODO: 아래 코드 뜻 해석하기
                if reg_idx.search(line): ## 1
                    continue
                
                elif line[:2]=='##' and not reg_label.search(line): # raw_text
                    token_count += 1
                    
                    raw_text = line[3:]
                    tokenized_text, start_index = transform_source_fn(raw_text)
                    # TODO: tokenizer의 method 정리하기
                    cls_sep_idx = tokenizer.token_to_cls_sep_idx(tokenized_text)
                    save_token_list.append(cls_sep_idx)
                    
                elif line[:2]=='##' and reg_label.search(line): # label_text
                    ner_count += 1
                    assert token_count==ner_count
                    
                    label_text = line[3:]
                    ner_tag = transform_target_fn(label_text, tokenized_text, start_index)
                    cls_sep_ner_idx_tag = ner_tag_to_idx(ner_tag)
                    
                    assert len(cls_sep_idx)==len(cls_sep_ner_idx_tag)
                    save_ner_list.append(cls_sep_ner_idx_tag)
                    

    # Save processed data to .pt files
    torch.save(save_token_list, './data/processed_data/{}_token_idx.pt'.format(m))
    torch.save(save_ner_list, './data/processed_data/{}_ner_idx.pt'.format(m))
    print('{} files saved to ./data/processed_data/{}_token_idx.pt'.format(m, m))