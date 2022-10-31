import cn2an
from thefuzz import fuzz
import re
import pandas as pd

# ssc code
from ssc.src.similarity import *
from ssc.src.soundshapecode import ssc
ssc.getHanziStrokesDict()
ssc.getHanziStructureDict()
ssc.getHanziSSCDict()

CN_UNIT = {
        '十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000, '仟': 1000, '万': 10000, '萬': 10000,
        '亿': 100000000, '億': 100000000, '兆': 1000000000000
}

def get_camr_dict(path='../../Chinese-AMR/tools/camr_pred_dict.xls'):
    df = pd.read_excel(path, header=None, dtype=str,  keep_default_na=False)
    camr_dict = {}
    concept_list = []

    for c, i, e in zip(df[0], df[1], df[2]):
        if(c == ''):
            continue
        if i != '' and i != 'xx':
            camr_dict[c+'-'+i] = e
        elif i == 'xx':
            camr_dict[c] = e
        
        if c not in concept_list:
            concept_list.append(c)

    return camr_dict, concept_list

def compute_similarity(str1, str2, ssc_encode_way = 'ALL'):
    chi_word1_ssc = ssc.getSSC(str1, ssc_encode_way)
    chi_word2_ssc = ssc.getSSC(str2, ssc_encode_way)
    res = compute_similarity_list(chi_word1_ssc, chi_word2_ssc, 'ALL')
    return res

def get_similar_concept(word, concept_list):
    concept = word
    max_sim = 0
    for i in concept_list:
        sim = compute_similarity(word, i)
        if sim > max_sim:
            max_sim = sim
            concept = i
    return (concept, max_sim)

def get_window(line, keywords, window_size=3, num=1):
    """
    获取当前行的第num个keywords的window,window_size固定,长度不足的情况用'#'补全
    example:
        line:"对人不对事"
        keywords: "对"
        num: 2
        代表这句话中第二个“对”
    """
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|？|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|【|】|·|！| |…|（|）'
    subsentences = re.split(pattern, line)
    match_num = 1
    for i in subsentences:
        subsentence = i  # 针对情况：'对：对事不对人'一个短句里出现两次关键词
        if keywords in subsentence:
            while(subsentence.find(keywords) != -1 and match_num < num):
                subsentence=subsentence.replace(keywords, '', 1)
                match_num += 1
            if(subsentence.find(keywords) == -1):
                continue

            start = subsentence.find(keywords) + len(i)-len(subsentence) - window_size
            end = start + 2*window_size + len(keywords)

            if(start >= 0 and len(i) > end):
                return i[start:end]

            str_return = i[0 if start < 0 else start:end]
            if(start < 0):
                str_return = '#'*(-start) + str_return
            if(end >= len(i)):
                str_return =  str_return + '#'*(end - len(i))
            return str_return
    return keywords

def get_window_token(line, keyword, window_size=3, num=1):
    """
    获取当前行的第num个keywords的window,window_size固定,长度不足的情况用'#'补全
    一个词占一位
    """
    sent_list = line.split(' ')
    match_num = 1
    sent_list_copy = sent_list.copy()

    while(keyword in sent_list_copy and match_num < num):
        sent_list_copy.remove(keyword)
        match_num += 1

    assert match_num == num and keyword in sent_list_copy

    key_index = sent_list_copy.index(keyword) + match_num - 1
    start = key_index - window_size
    end = key_index + window_size

    if(start >= 0 and len(sent_list) > end):
        return ''.join(sent_list[start:end+1])

    sent_return = sent_list[(0 if start < 0 else start):end]
    if(start < 0):
        sent_return = ['#' for i in range(-start)] + sent_return
    if(end >= len(sent_list)):
        sent_return =  sent_return + ['#' for i in range(end-len(sent_list))]
    
    return ''.join(sent_return)


def frac_val1(train_sense):
    val = []
    for i in train_sense:
        val.append(i[1])
    return val[0] / sum(val)

def metric(train_sense, temperature=0):
    return frac_val1(train_sense) #/ (len(train_sense)+temperature) 

def filter_num(string):
    restr = ''
    for i in string:
        if i >= '0' and i <= '9' or i == '.':
            restr += i
    return restr

def compute_word_sense(path):
    with open(path,"r") as f:
        lines = [eval(i.strip()) for i in f.readlines()]
    align_ws = {}
    align_line = {}
    line_cnt = 0
    for i in lines:
        line_cnt += 1
        for j in i:
            if j[0] not in align_ws.keys():
                align_ws[j[0]] = {j[1]:1}
                align_line[j[0]] = {j[1]:[line_cnt]}
            else:
                if j[1] not in align_ws[j[0]].keys():
                    align_ws[j[0]][j[1]] = 1
                    align_line[j[0]][j[1]] = [line_cnt]
                else:
                    align_ws[j[0]][j[1]] += 1
                    align_line[j[0]][j[1]].append(line_cnt)
    return align_ws, align_line


def compute_word_context(sense_path, sent_path):
    """
    收集每个概念对应的转换和上下文
    sense_path: .p_align文件
    sent_path: .sent文件
    """
    with open(sense_path,"r") as f:
        lines = [eval(i.strip()) for i in f.readlines()]
    with open(sent_path,"r") as f:
        sentences = f.readlines()
    def extract_subsent(line, keywords, num=1):
        sentence = sentences[line][:-1].replace(" ", "")
        return get_window(sentence, keywords, window_size=4, num=num)
    align_ws = {}
    align_sent = {}
    line_cnt = -1
    for i in lines:
        line_cnt += 1
        currence = {}
        for j in i:
            if j[0] not in currence.keys():
                currence[j[0]] = 1
            else:
                currence[j[0]] += 1
            if j[0] not in align_ws.keys():
                align_ws[j[0]] = {j[1]:1}
                align_sent[j[0]] = {j[1]:[extract_subsent(line_cnt, j[0], currence[j[0]])]}
            else:
                if j[1] not in align_ws[j[0]].keys():
                    align_ws[j[0]][j[1]] = 1
                    align_sent[j[0]][j[1]] = [extract_subsent(line_cnt, j[0], currence[j[0]])]
                else:
                    align_ws[j[0]][j[1]] += 1
                    align_sent[j[0]][j[1]].append(extract_subsent(line_cnt, j[0], currence[j[0]]))
    return align_ws, align_sent

#compute baseline acc for wsd and none wsd : concept normalization
total_num=0
match_num=0

none_wsd=0
match_none_wsd=0

align_ws_train, align_sent_train = compute_word_context("./wsd_dataset/train_clean.p_align", "./wsd_dataset/train.sent")
align_ws_dev, align_sent_dev = compute_word_context("./wsd_dataset/dev.p_align", "./wsd_dataset/dev.sent")
concept_dict, concept_list = get_camr_dict()
# 统计错误情况
undeclared_case = []
undeclared_match = 0
undeclared_unmatch = 0
declared_case = []
declared_match = 0
declared_unmatch = 0





def normalize_concept(i):
    i = i.replace("两", "二")
    if i in align_ws_train.keys():
        train_sense = sorted(align_ws_train[i].items(),key=lambda x:-x[1])
        if metric(train_sense, 1) <= 0.8:
            declared_case.append(i)
            return -1
    else:
        if(i != cn2an.transform(i)):
            train_sense_str = cn2an.transform(i)
        elif(i in CN_UNIT.keys()):
            train_sense_str = str(CN_UNIT[i])
        elif(i[-1] in CN_UNIT.keys()):
            train_sense_str = str(round(float(cn2an.transform(i[:-1])) * CN_UNIT[i[-1]]))
        else:
            train_sense_str = i
        
        if(re.findall('\d+',train_sense_str) != []):
            train_sense_str = filter_num(train_sense_str)
            train_sense = [(train_sense_str, 1)]
        else:
            if i in concept_list:
                train_sense = [(i + '-01', 1)]
            else:
                train_sense_str, similarity = get_similar_concept(i, concept_list)
                if(similarity < 0.2):
                    train_sense = [(i, similarity)]
                else:
                    train_sense = [(train_sense_str + '-01', similarity)]
    return train_sense

def fuzzy_match_avg(i, sentence, align_sent_train):
    concept = None
    sum_scores = 0
    concept_num = 0
    avg_scores = {}
    for key in align_sent_train[i].keys():
        for sentence_train in align_sent_train[i][key]:
            sum_scores += fuzz.ratio(sentence, sentence_train)
            concept_num += 1
        avg_scores[key] = sum_scores / concept_num
    concept = max(avg_scores, key=lambda i:avg_scores[i])
    return concept

def fuzzy_match(i, sentence, align_sent_train):
    concept = None
    max_scores = 0
    for key in align_sent_train[i].keys():
        for sentence_train in align_sent_train[i][key]:
            if(fuzz.ratio(sentence, sentence_train) > max_scores):
                max_scores = fuzz.ratio(sentence, sentence_train)
                concept = key
    return concept

def normalize_concept_sentence(i, sentence, concepts, align_sent_train,num=1):
    """
    * normalize_concept_sentence('到','欢迎各位领导到我公司视察',concepts, align_sent_train, 1) 
    * return '到-01'
    同时需要训练集的所有上下文字典: align_sent_train 通过改后的compute_word_context函数获得
    新增一个参数concepts: camr_pred_dict.xls中第一列所有的词 通过get_camr_dict函数获得
    """
    i = i.replace("两", "二")
    if i in align_ws_train.keys():
        train_sense = sorted(align_ws_train[i].items(),key=lambda x:-x[1])
        if metric(train_sense, 1) <= 0.8:
            context = get_window(sentence, i, window_size=4, num=num)
            return fuzzy_match(i, context, align_sent_train)
    else:
        if(i != cn2an.transform(i)):
            train_sense_str = cn2an.transform(i)
        elif(i in CN_UNIT.keys()):
            train_sense_str = str(CN_UNIT[i])
        elif(i[-1] in CN_UNIT.keys()):
            train_sense_str = str(round(float(cn2an.transform(i[:-1])) * CN_UNIT[i[-1]]))
        else:
            train_sense_str = i
        
        if(re.findall('\d+',train_sense_str) != []):
            train_sense_str = filter_num(train_sense_str)
            train_sense = [(train_sense_str, 1)]
        else:
            if i in concepts:
                train_sense = [(i + '-01', 1)]
            else:
                train_sense_str, similarity = get_similar_concept(i, concepts)
                if(similarity < 0.2):
                    train_sense = [(i, 1)]
                else:
                    train_sense = [(train_sense_str + '-01', 1)]
    return train_sense

camrs,concepts=get_camr_dict()

sense = normalize_concept_sentence('视察','欢迎各位领导到我公司视察',concepts, align_sent_train, 1) 

print(sense)

# for i,j in align_ws_dev.items():
#     train_sense = normalize_concept(i)
#     if(train_sense != -1):
#         for sense,num in j.items():
#             if "-" in sense:
#                 total_num += num
#                 if train_sense!=None and sense == train_sense[0][0]:
#                     match_num += num
#                     if i in declared_case:
#                         declared_match += num
#                 else:
#                     if i in declared_case:
#                         declared_unmatch += num
#                         #print(i, sense, train_sense[0][0], num, sep=' ')
#             else:
#                 none_wsd += num
#                 if train_sense!=None and sense == train_sense[0][0]:
#                     match_none_wsd += num
#                     if i in declared_case:
#                         declared_match += num
#                 else:
#                     if i in declared_case:
#                         declared_unmatch += num
#     else:
#         for key in align_sent_dev[i]:
#             for sentence in align_sent_dev[i][key]:
#                 if "-" in key:
#                     total_num += 1
#                 else:
#                     none_wsd += 1
#                 if fuzzy_match(i, sentence, align_sent_train) != key:
#                     declared_unmatch += 1
#                 else:
#                     declared_match += 1
#                     if "-" in key:
#                         match_num += 1
#                     else:
#                         match_none_wsd += 1


# print("metric下错误的概念：", declared_unmatch)
# print("metric下正确的概念：", declared_match)
# print("total pred:%d, none pred:%d, wsd acc: %.3f, none wsd acc: %.3f, acc: %.3f"%(total_num,none_wsd,match_num/total_num,match_none_wsd/none_wsd,(match_num+match_none_wsd)/(total_num+none_wsd)))
