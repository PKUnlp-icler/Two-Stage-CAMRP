#sid	nid1	concept1	coref1	rel	rid	ralign	nid2	concept2	coref2
from ast import Break
import string
import numpy as np
class camr_tuple:
    def __init__(self):
        self.sid = "-"
        self.nid1 = "-"
        self.concept1 = "-"
        self.coref1 = "-"
        self.rel = "-"
        self.rid = "-"
        self.ralign = "-"
        self.nid2 = "-"
        self.concept2 = "-"
        self.coref2 = "-"
    
    def load_from_str(self, str):
        str = str.strip()
        str = str.split('\t')
        self.sid = str[0]
        self.nid1 = str[1]
        self.concept1 = str[2]
        self.coref1 = str[3]
        self.rel = str[4]
        self.rid = str[5]
        self.ralign = str[6]
        self.nid2 = str[7]
        self.concept2 = str[8]
        self.coref2 = str[9]
    
    def __str__(self):
        return '\t'.join([self.sid, self.nid1, self.concept1, self.coref1, self.rel, self.rid, self.ralign, self.nid2, self.concept2, self.coref2])

def load_tuple_file(file_path):
    tuple_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # if start with number
            if line[0].isdigit():
                tuple_list.append(camr_tuple())
                tuple_list[-1].load_from_str(line)
    return tuple_list

def compute_vocab(tuple_list):
    concepts={}
    relations={}
    nodes={}
    ralign={}
    for tuple in tuple_list:
        if tuple.concept1 not in concepts:
            concepts[tuple.concept1] = 1
        else:
            concepts[tuple.concept1] += 1
        if tuple.concept2 not in concepts:
            concepts[tuple.concept2] = 1
        else:
            concepts[tuple.concept2] += 1
        if tuple.rel not in relations:
            relations[tuple.rel] = 1
        else:
            relations[tuple.rel] += 1
        if tuple.ralign not in ralign:
            ralign[tuple.ralign] = 1
        else:
            ralign[tuple.ralign] += 1
        if tuple.nid1 not in nodes:
            nodes[tuple.nid1] = 1
        else:
            nodes[tuple.nid1] += 1
        if tuple.nid2 not in nodes:
            nodes[tuple.nid2] = 1
        else:
            nodes[tuple.nid2] += 1

    # sort vocabs
    concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
    relations = sorted(relations.items(), key=lambda x: x[1], reverse=True)
    ralign = sorted(ralign.items(), key=lambda x: x[1], reverse=True)
    nodes = sorted(nodes.items(), key=lambda x: x[1], reverse=True)

    predicates = [i for i in concepts if "-" in i[0] and i[0][-1].isdigit()] 

    # if predicate is English 
    english_predicates = [i for i in concepts if i[0][0] in list(string.ascii_lowercase)]
    
    return concepts, relations, ralign, nodes,predicates,english_predicates


def write_vocabs(vocabs, file_path):
    with open(file_path, 'w') as f:
        for vocab in vocabs:
            f.write(vocab[0] + ' ' + str(vocab[1]) + '\n')

def tuple_group_by_sid(tuple_list):
    sid_group = {}
    for tuple in tuple_list:
        if tuple.sid not in sid_group:
            sid_group[tuple.sid] = []
        sid_group[tuple.sid].append(tuple)
    return sid_group

def get_instance_relation_anchor(tuples,maxlen):
    anchor_instance={}
    none_anchor_instance={}
    relation=[]
    word_relation=[]
    ralign_anchor={}

    anchor_instance_no_r={}

    for tuple in tuples:
        if "_" in tuple.nid1 or int(tuple.nid1[1:])<=maxlen:
            if tuple.nid1 not in anchor_instance:
                anchor_instance[tuple.nid1]=tuple.concept1
            if tuple.nid1 not in anchor_instance_no_r:
                anchor_instance_no_r[tuple.nid1]=tuple.concept1
        if "_" in tuple.nid2 or int(tuple.nid2[1:])<=maxlen:
            if tuple.nid2 not in anchor_instance:
                anchor_instance[tuple.nid2]=tuple.concept2
            if tuple.nid2 not in anchor_instance_no_r:
                anchor_instance_no_r[tuple.nid2]=tuple.concept2
        
        if "_" not in tuple.nid1 and int(tuple.nid1[1:])>maxlen:
            if tuple.nid1 not in none_anchor_instance:
                none_anchor_instance[tuple.nid1]=tuple.concept1

        if "_" not in tuple.nid2 and int(tuple.nid2[1:])>maxlen:
            if tuple.nid2 not in none_anchor_instance:
                none_anchor_instance[tuple.nid2]=tuple.concept2



        #get word relation
        if tuple.rid!="-" and tuple.ralign!="-":
            anchor_instance[tuple.rid]=tuple.ralign  #修改0811
            ralign_anchor[tuple.rid]=tuple.ralign
            word_relation.append([tuple.nid1,tuple.nid2,tuple.rel,tuple.rid,tuple.ralign])
        else:
            #get relation
            relation.append([tuple.nid1,tuple.rel,tuple.nid2])

        

    
    return {'anchor_instance':anchor_instance,'none_anchor_instance':none_anchor_instance,'relation':relation,'word_relation':word_relation,'ralign_anchor':ralign_anchor,'anchor_instance_no_r':anchor_instance_no_r}




def write_tagging_dataset_extra_nodes_2classes(tuple_list,word_list,tag_path,lengh_path,out_path=""):
    '''
    generate labels for each word
    设置n个类
    ：0-未触发生成节点 1-出发生成节点
    '''

    lp={}
    with open(lengh_path,"r") as f:
        lengths=f.readlines()
        for i in lengths:
            i = i.strip()
            lp[i.split("\t")[0]]=int(i.split("\t")[1])
    
    tags_index={}
    with open(tag_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            tags_index[j.strip()]=i+1
    
    print(tags_index)



    num_extra_nodes=0
    num_tagged_nodes=0
    num_extra_length_nodes=0

    untag_nodes={}
    f_extra_nodes_tag = open(out_path+".extra_nodes.tag","w")
    for w_id,(t_id,t_tuples) in zip(word_list,tuple_list.items()):
        t_dict = get_instance_relation_anchor(t_tuples,lp[t_id])
        num_extra_nodes+=len(t_dict['none_anchor_instance'])

        tagging_label=["0"]*len(w_id)
        for x_id,x_name in t_dict['none_anchor_instance'].items():
            for i in t_dict['relation']:
                relation_x_id,relation,relationx2_id = i
                if relation_x_id==x_id:
                    trigger_index = int(relationx2_id.split("_")[0][1:])-1
                    if trigger_index >= len(w_id):
                        continue
                    else:
                        if x_name in tags_index.keys():
                            tagging_label[trigger_index] = str(tags_index[x_name])
                            num_tagged_nodes+=1
                        break
                

        f_extra_nodes_tag.write(" ".join(tagging_label)+"\n")
    
    print(untag_nodes)
    print(num_extra_nodes,num_tagged_nodes)



def write_ralign_classification_dataset(tuple_list,word_list,relation_path,lengh_path,out_path="",reverse_of=True):
    lp={}
    with open(lengh_path,"r") as f:
        lengths=f.readlines()
        for i in lengths:
            i = i.strip()
            lp[i.split("\t")[0]]=int(i.split("\t")[1])
    
    relation_index={}
    with open(relation_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            relation_index[j.strip().split()[0]]=i+1
    
    num_relations=0
    num_tagged_relations=0

    f_relations_nodes = open(out_path+".4level.ralign.relations_nodes","w")
    f_relations = open(out_path+".4level.ralign.relations","w")
    f_relations_literal = open(out_path+"..4levelralign.relations.literal","w")



    for w_id,(t_id,t_tuples) in zip(word_list,tuple_list.items()):
        t_dict = get_instance_relation_anchor(t_tuples,lp[t_id])

        # print(t_dict['anchor_instance'])
        # print(t_dict['none_anchor_instance'])

        literal_relations=[]

        total_input_nodes = [(i,j) for i,j in t_dict['anchor_instance'].items()] + [(i,j) for i,j in t_dict['none_anchor_instance'].items()]
        sorted_nodes = sorted(total_input_nodes,key=lambda x:int(x[0].split("_")[0][1:]))
        
        total_relations = np.zeros([len(total_input_nodes),len(total_input_nodes)])
        total_relations_no_r = np.zeros([len(total_input_nodes),len(total_input_nodes)])

        x_ids = [i[0] for i in sorted_nodes]
                
        
        for i in t_dict['word_relation']:
            literal_relations.append([i[0],i[2],i[3],i[4],i[1]])

            start_index = x_ids.index(i[0])
            end_index = x_ids.index(i[1])

            if reverse_of and i[2].endswith('-of') and i[2] != 'consist-of':
                relation_label = relation_index[i[2][:-3]]
            else:
                relation_label = relation_index[i[2]]

            end_index2 = x_ids.index(i[3])
            total_relations[start_index][end_index] = relation_label
            total_relations[start_index][end_index2] = relation_label


        total_input_nodes_rename_extra = [(i,j) for i,j in t_dict['anchor_instance'].items()] + [(i.replace("x","b"),j) for i,j in t_dict['none_anchor_instance'].items()]
        total_input_nodes_no_r_rename_extra = [(i,j) for i,j in t_dict['anchor_instance_no_r'].items()] + [(i.replace("x","b"),j) for i,j in t_dict['none_anchor_instance'].items()]
        
        
        sorted_nodes_rename_extra = sorted(total_input_nodes_rename_extra,key=lambda x:int(x[0].split("_")[0][1:]))

        f_relations_nodes.write(str(sorted_nodes_rename_extra)+"\n")
        f_relations.write(str(total_relations.tolist())+"\n")
        f_relations_literal.write(str(t_id)+"\t"+str(literal_relations)+"\n")

    


def write_4_level_relation_classification_dataset(tuple_list,word_list,relation_path,lengh_path,out_path="",reverse_of=True):
    lp={}
    with open(lengh_path,"r") as f:
        lengths=f.readlines()
        for i in lengths:
            i = i.strip()
            lp[i.split("\t")[0]]=int(i.split("\t")[1])
    
    relation_index={}
    with open(relation_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            relation_index[j.strip().split()[0]]=i+1
    
    print(relation_index)



    num_relations=0
    num_tagged_relations=0

    f_relations_nodes = open(out_path+".4level.relations_nodes","w")
    # f_relation_nodes_nl = open(out_path+".4level.relations_nodes_nl","w")


    f_relations = open(out_path+".4level.relations","w")
    f_relations_no_r = open(out_path+".4level.relations.no_r","w")
    f_relations_literal = open(out_path+".4level.relations.literal","w")
    f_relations_nodes_no_ralign =  open(out_path+".4level.relations_nodes_no_r","w")


    for w_id,(t_id,t_tuples) in zip(word_list,tuple_list.items()):
        t_dict = get_instance_relation_anchor(t_tuples,lp[t_id])

        # print(t_dict['anchor_instance'])
        # print(t_dict['none_anchor_instance'])

        literal_relations=[]

        total_input_nodes = [(i,j) for i,j in t_dict['anchor_instance'].items()] + [(i,j) for i,j in t_dict['none_anchor_instance'].items()]
        sorted_nodes = sorted(total_input_nodes,key=lambda x:int(x[0].split("_")[0][1:]))
        
        total_relations = np.zeros([len(total_input_nodes),len(total_input_nodes)])
        total_relations_no_r = np.zeros([len(total_input_nodes),len(total_input_nodes)])

        x_ids = [i[0] for i in sorted_nodes]

        for i in t_dict['relation']:
            literal_relations.append(i)
            start_index = x_ids.index(i[0])
            end_index = x_ids.index(i[2])
            try:
                if reverse_of and i[1].endswith('-of') and i[1] != 'consist-of':
                    relation_label = relation_index[i[1][:-3]]
                else:
                    relation_label = relation_index[i[1]]
                
                total_relations[start_index][end_index] = relation_label
                total_relations_no_r[start_index][end_index] = relation_label
            except KeyError as e:
                print(e)
                
        
        for i in t_dict['word_relation']:
            literal_relations.append([i[0],i[2],i[3],i[4],i[1]])

            start_index = x_ids.index(i[0])
            end_index = x_ids.index(i[1])

            if reverse_of and i[2].endswith('-of') and i[2] != 'consist-of':
                relation_label = relation_index[i[2][:-3]]
            else:
                relation_label = relation_index[i[2]]

            end_index2 = x_ids.index(i[3])
            total_relations[start_index][end_index] = relation_label
            total_relations_no_r[start_index][end_index] = relation_label  # don't label the ralign word

            total_relations[start_index][end_index2] = relation_label


        total_input_nodes_rename_extra = [(i,j) for i,j in t_dict['anchor_instance'].items()] + [(i.replace("x","b"),j) for i,j in t_dict['none_anchor_instance'].items()]
        total_input_nodes_no_r_rename_extra = [(i,j) for i,j in t_dict['anchor_instance_no_r'].items()] + [(i.replace("x","b"),j) for i,j in t_dict['none_anchor_instance'].items()]
        
        
        sorted_nodes_rename_extra = sorted(total_input_nodes_rename_extra,key=lambda x:int(x[0].split("_")[0][1:]))

        # import pdb
        # pdb.set_trace()

        
        sorted_nodes_rename_extra_nl_version = []

        word_id_dict = {i.split("_")[0]:i.split("_")[1] for i in w_id}

        for index,concept in sorted_nodes_rename_extra:
            if "_" not in index and "b" not in index and "x0" != index and index in word_id_dict.keys() :
                sorted_nodes_rename_extra_nl_version.append((index,word_id_dict[index]))
            else:
                sorted_nodes_rename_extra_nl_version.append((index,concept))
        # natural language version

        
        sorted_nodes_rename_extra_no_r = sorted(total_input_nodes_no_r_rename_extra,key=lambda x:int(x[0].split("_")[0][1:]))

        #f_relation_nodes_nl.write(str(sorted_nodes_rename_extra_nl_version)+"\n")
        f_relations_nodes.write(str(sorted_nodes_rename_extra)+"\n")
        f_relations.write(str(total_relations.tolist())+"\n")
        f_relations_no_r.write(str(total_relations_no_r.tolist())+"\n")
        f_relations_literal.write(str(t_id)+"\t"+str(literal_relations)+"\n")
        f_relations_nodes_no_ralign.write(str(sorted_nodes_rename_extra_no_r)+"\n")





def write_tagging_dataset_v4_8classes(tuple_list,word_list,lengh_path,out_path=""):
    '''
    generate labels for each word
    设置8个类
    ：B-单个词对应概念 1
    ：B-需要对齐的虚词 7
    ：B-连续词 2， I-连续词 3
    ：B-非连续词 4， I-非连续词 5
    ：B-拆分词 6
    ：O 无关词 0
    '''

    lp={}
    with open(lengh_path,"r") as f:
        lengths=f.readlines()
        for i in lengths:
            i = i.strip()
            lp[i.split("\t")[0]]=int(i.split("\t")[1])

    total_tags=0
    num_node=0
    num_node_need_wsd=0
    is_continued_multi=0
    is_uncontinued_multi=0
    is_cut_word=0

    f_tag = open(out_path+".tag","w")
    for w_id,(t_id,t_tuples) in zip(word_list,tuple_list.items()):
        t_dict = get_instance_relation_anchor(t_tuples,lp[t_id])
        index_list = []
        node_list = []
        tagging_label = []
        word_transform = []
        word_transform_tags = []
        word_transform_labels= []

        #NEW
        no_ralign_index_list = []
        for i in sorted(t_dict['anchor_instance_no_r'].items(), key=lambda x: int(x[0][1:].split("_")[0]))[1:]:
            no_ralign_index_list.append(i[0])

        for i in sorted(t_dict['anchor_instance'].items(), key=lambda x: int(x[0][1:].split("_")[0]))[1:]:
            index_list.append(i[0])
            node_list.append(i[1])

        for i,j in enumerate(w_id):
            total_tags+=1
            if j.split("_")[0] in index_list and j.split("_")[0] in no_ralign_index_list:
                tagging_label.append("1")
                num_node+=1
                # concept maybe does not equal the literal word
                # 通过 -> 通过-01 , 一千 -> 1000, '今后' -> 'temporal-quantity'
            elif j.split("_")[0] in index_list and j.split("_")[0] not in no_ralign_index_list:
                tagging_label.append("7")
                num_node+=1

            else:
                tagging_label.append("0")
        
        # detect continued Nodes and incontiuned Nodes, tagged with 1,2 and 3,4

        # eg: 明田 公司 -> 2 3   , 在 ..... 上 -> 4 5
        # print(w_id,index_list,node_list)

        # print(tagging_label)
        
        for i,j in zip(index_list,node_list):
            if "_" in i:
                xs = i.split("_")
                is_multiword=1
                is_continued = 1
                is_split_word=0
                for q in xs:
                    if "x" not in q:
                        # not a multi word node, is a word split nodes
                        is_multiword=0
                        is_split_word=1
                        break
                if is_multiword:
                    word_ids = [int(i[1:]) for i in xs]
                    for c_index,c_id in enumerate(word_ids[1:]):
                        if c_id-1 != word_ids[c_index]:
                            is_continued = 0
                    
                    # continued multiword

                    
                    
                    for index,i in enumerate(word_ids):
                        if is_continued:
                            if index==0:
                                tagging_label[i-1]="2"
                            else:
                                tagging_label[i-1]="3"
                            is_continued_multi+=1
                        else:
                            if index==0:
                                tagging_label[i-1]="4"
                            else:
                                tagging_label[i-1]="5"
                            is_uncontinued_multi+=1
                    
                    # split_words
                else:
                    #print(xs,"split")
                    node_index = int(xs[0][1:])
                    tagging_label[node_index-1]="6"
                    is_cut_word +=1
        

        f_tag.write(" ".join(tagging_label)+"\n")

    
    print(num_node,num_node_need_wsd,is_continued_multi,is_uncontinued_multi,is_cut_word)


        
        
    


def write_tagging_dataset_concept_norm(tuple_list,word_list,lengh_path,out_path=""):
    '''
    generate labels for each word
    设置6个类
    ：B-连续词 ， I-连续词
    ：B-非连续词， I-非连续词
    ：B-拆分词
    ：O 无关词
    '''

    lp={}
    with open(lengh_path,"r") as f:
        lengths=f.readlines()
        for i in lengths:
            i = i.strip()
            lp[i.split("\t")[0]]=int(i.split("\t")[1])

    total_tags=0
    num_node=0
    num_node_need_wsd=0
    is_continued_multi=0
    is_uncontinued_multi=0
    is_cut_word=0


    # f_tag = open(out_path+".tag","w")
    f_transform_tag = open(out_path+".p_transform","w")
    f_transform_label = open(out_path+".p_transform_tag","w")



    
    for w_id,(t_id,t_tuples) in zip(word_list,tuple_list.items()):
        t_dict = get_instance_relation_anchor(t_tuples,lp[t_id])
        index_list = []
        node_list = []
        tagging_label = []
        word_transform = []
        word_transform_tags = []
        word_transform_labels= []
        for i in sorted(t_dict['anchor_instance'].items(), key=lambda x: int(x[0][1:].split("_")[0]))[1:]:
            index_list.append(i[0])
            node_list.append(i[1])

        

        for i,j in enumerate(w_id):
            total_tags+=1
            if j.split("_")[0] in index_list:
                tagging_label.append("1")
                num_node+=1
                # concept does not equal the literal word
                # 通过 -> 通过-01 , 一千 -> 1000, '今后' -> 'temporal-quantity'
            else:
                tagging_label.append("0")
            
            if j.split("_")[0] in index_list:
                node_index_in_list = index_list.index(j.split("_")[0])
                if j.split("_")[1]!=node_list[node_index_in_list]:
                    word_transform.append([j.split("_")[1],node_list[node_index_in_list]])
                    word_transform_tags.append(node_list[node_index_in_list])
                    word_transform_labels.append("1")
                    num_node_need_wsd+=1
                else:
                    word_transform_tags.append("NONE")
                    word_transform_labels.append("0")
            else:
                word_transform_tags.append("NONE")
                word_transform_labels.append("0")
        
        # detect continued Nodes and incontiuned Nodes, tagged with 1,2 and 3,4

        # eg: 明田 公司 -> 1 2   , 在 ..... 上 -> 3 4


        # print(w_id,index_list,node_list)

        # print(tagging_label)
        
        for i,j in zip(index_list,node_list):
            if "_" in i:
                xs = i.split("_")
                is_multiword=1
                is_continued = 1
                is_split_word=0
                for q in xs:
                    if "x" not in q:
                        # not a multi word node, is a word split nodes
                        is_multiword=0
                        is_split_word=1
                        break
                if is_multiword:
                    word_ids = [int(i[1:]) for i in xs]
                    for c_index,c_id in enumerate(word_ids[1:]):
                        if c_id-1 != word_ids[c_index]:
                            is_continued = 0
                    
                    # continued multiword

                    
                    
                    for index,i in enumerate(word_ids):
                        if is_continued:
                            if index==0:
                                tagging_label[i-1]="1"
                            else:
                                tagging_label[i-1]="2"
                            is_continued_multi+=1
                        else:
                            if index==0:
                                tagging_label[i-1]="3"
                            else:
                                tagging_label[i-1]="4"
                            is_uncontinued_multi+=1
                    
                    

                    # split_words
                else:
                    #print(xs,"split")
                    node_index = int(xs[0][1:])
                    tagging_label[node_index-1]="5"
                    is_cut_word +=1
        
        tagged_words = [i+" "+j for i,j in zip(tagging_label,w_id)]

        assert len(tagging_label)==len(w_id)==len(word_transform_tags)
        # f_tag.write(" ".join(tagging_label)+"\n")
        f_transform_tag.write(str(word_transform_tags)+"\n")
        f_transform_label.write(" ".join(word_transform_labels)+"\n")
    
    print(num_node,num_node_need_wsd,is_continued_multi,is_uncontinued_multi,is_cut_word)
        


def write_ralign_classification_dataset(tuple_list,word_list,relation_path,lengh_path,out_path="",reverse_of=True):
    lp={}
    with open(lengh_path,"r") as f:
        lengths=f.readlines()
        for i in lengths:
            i = i.strip()
            lp[i.split("\t")[0]]=int(i.split("\t")[1])
    
    relation_index={}
    with open(relation_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            relation_index[j.strip().split()[0]]=i+1
    
    num_relations=0
    num_tagged_relations=0

    f_relations_nodes = open(out_path+".4level.ralign.relations_nodes","w")
    f_relations = open(out_path+".4level.ralign.relations","w")
    f_relations_literal = open(out_path+"..4levelralign.relations.literal","w")



    for w_id,(t_id,t_tuples) in zip(word_list,tuple_list.items()):
        t_dict = get_instance_relation_anchor(t_tuples,lp[t_id])

        # print(t_dict['anchor_instance'])
        # print(t_dict['none_anchor_instance'])

        literal_relations=[]

        total_input_nodes = [(i,j) for i,j in t_dict['anchor_instance'].items()] + [(i,j) for i,j in t_dict['none_anchor_instance'].items()]
        sorted_nodes = sorted(total_input_nodes,key=lambda x:int(x[0].split("_")[0][1:]))
        
        total_relations = np.zeros([len(total_input_nodes),len(total_input_nodes)])
        total_relations_no_r = np.zeros([len(total_input_nodes),len(total_input_nodes)])

        x_ids = [i[0] for i in sorted_nodes]
                
        
        for i in t_dict['word_relation']:
            literal_relations.append([i[0],i[2],i[3],i[4],i[1]])

            start_index = x_ids.index(i[0])
            end_index = x_ids.index(i[1])

            if reverse_of and i[2].endswith('-of') and i[2] != 'consist-of':
                relation_label = relation_index[i[2][:-3]]
            else:
                relation_label = relation_index[i[2]]

            end_index2 = x_ids.index(i[3])
            total_relations[start_index][end_index] = relation_label
            total_relations[start_index][end_index2] = relation_label


        total_input_nodes_rename_extra = [(i,j) for i,j in t_dict['anchor_instance'].items()] + [(i.replace("x","b"),j) for i,j in t_dict['none_anchor_instance'].items()]
        total_input_nodes_no_r_rename_extra = [(i,j) for i,j in t_dict['anchor_instance_no_r'].items()] + [(i.replace("x","b"),j) for i,j in t_dict['none_anchor_instance'].items()]
        
        
        sorted_nodes_rename_extra = sorted(total_input_nodes_rename_extra,key=lambda x:int(x[0].split("_")[0][1:]))

        f_relations_nodes.write(str(sorted_nodes_rename_extra)+"\n")
        f_relations.write(str(total_relations.tolist())+"\n")
        f_relations_literal.write(str(t_id)+"\t"+str(literal_relations)+"\n")



def get_word_id(input_file):
    out=[]
    with open(input_file,"r") as f:
        lines=f.readlines()
        for line in lines:
            if line.startswith("# ::wid"):
                out.append(line.split("# ::wid")[1].strip().split(" "))
    return out
        

        
def get_input(input_file,out_put_file_sent,out_put_file_wid):

    out_sent=open(out_put_file_sent,"w")
    out_wid=open(out_put_file_wid,"w")
    with open(input_file,"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip()
            if line.startswith("# ::snt"):
                out_sent.write(line.split("# ::snt")[1]+"\n")
            if line.startswith("# ::wid"):
                out_wid.write(line.split("# ::wid")[1]+"\n")
    out_sent.close()
    out_wid.close()

def construct_clean_dict(dict_path, nodes_path, clean_path):
    en_list = []
    with open(dict_path) as f:
        for line in f.readlines():
            en_list.append(line[:-1])

    with open(nodes_path) as f:
        nodes = f.read()
    
    en_dict = {}
    label_cnt = 1
    #f = open(clean_path, 'w')
    for i in en_list:
        count = nodes.count(' '+i) + nodes.count(i+' ')
        if count >= 10:
            en_dict[i] = label_cnt
            label_cnt += 1
            #f.writelines(i + '\n')
    return en_dict


def load_dict(dict_path):
    en_dict = {}
    with open(dict_path) as f:
        for i, line in enumerate(f.readlines()):
            en_dict[line[:-1]] = i+1
    return en_dict
    

def load_tuples(tuples_path):
    tuples = []
    with open(tuples_path) as f:
        lines = f.readlines()
    
    tuple_sent = []
    line_dict = {}
    for line in lines[3:]:
        if line == '\n':
            tuples.append(tuple_sent)
            tuple_sent = []
            continue
        
        line_dict['节点编号1'] = line[:-1].split('\t')[1]
        line_dict['概念1'] = line[:-1].split('\t')[2]
        line_dict['节点编号2'] = line[:-1].split('\t')[7]
        line_dict['概念2'] = line[:-1].split('\t')[8]

        tuple_sent.append(line_dict.copy())

    return tuples

def find_pos(special_idx, tuples_line, sentence_len):
    pos = special_idx
    while(pos > sentence_len):
        next_hop_list = []
        for tuple_dict in tuples_line:
            if(int(tuple_dict['节点编号1'].split('_')[0][1:]) == pos):
                next_hop_list.append(int(tuple_dict['节点编号2'].split('_')[0][1:]))
        try:
            pos = min(next_hop_list)
        except:
            return -1
    return pos


def construct_tag_files(sent_path, en_dict, tuples, out_path):
    with open(sent_path) as f:
        sentences = f.readlines()

    f = open(out_path, 'w')

    line_num = 0
    for sentence in sentences:
        sentence_lst = sentence.strip().split()
        sentence_len = len(sentence_lst)
        tags = [0 for i in range(sentence_len)]
        
        tag_idx = [] # 保证每个非对齐概念被触发一次
        special_idxs = []
        for tuple_dict in tuples[line_num]:
            cur_idx = int(tuple_dict['节点编号1'].split('_')[0][1:])
            if cur_idx > sentence_len and cur_idx not in tag_idx:
                try:
                    tag = en_dict[tuple_dict['概念1']]
                except:
                    continue

                assert type(tag) == int

                ans = []
                for t in tuples[line_num]:
                    if cur_idx == int(t['节点编号1'].split('_')[0][1:]):
                        for i in t['节点编号2'].split('_'):
                            if i[0] == 'x' and int(i[1:]) <= sentence_len:
                                ans.append(int(i[1:]) - 1)
                
                tag_flag = False
                if ans != []:
                    for i in sorted(ans, reverse=True):
                        if tags[i] == 0:
                            tags[i] = tag
                            tag_idx.append(cur_idx)
                            tag_flag = True
                            break                   
                if tag_flag == False:
                    special_idxs.append((cur_idx, tag))
                    tag_idx.append(cur_idx)

        for i,t in special_idxs:
            pos = find_pos(i, tuples[line_num], sentence_len) - 1
            if pos < 0:
                continue
            for l in range(sentence_len - pos):
                if tags[l+pos] == 0:
                    tags[l+pos] = t
                    break

        f.writelines(' '.join([str(i) for i in tags]) + '\n')
        line_num += 1

    return

LENGTH_FILE="../Chinese-AMR-main/tools/max_len.txt"
dev_t_list = load_tuple_file("../datasets/camr_tuples/tuples_dev.txt")
dev_wid_list = get_word_id("../datasets/camr/camr_dev.txt")
dev_grouped_t = tuple_group_by_sid(dev_t_list)


train_t_list = load_tuple_file("../datasets/camr_tuples/tuples_train.txt")
train_wid_list = get_word_id("../datasets/camr/camr_train.txt")
train_grouped_t = tuple_group_by_sid(train_t_list)


#concepts, relations, ralign, nodes,predicates,eng_predicates = compute_vocab(t_list)
# write_vocabs(concepts,"vocabs_0615/concepts.txt")
# write_vocabs(relations,"vocabs_0615/relations.txt")
# write_vocabs(ralign,"vocabs_0615/ralign.txt")
# write_vocabs(nodes,"vocabs_0615/nodes.txt")
# write_vocabs(predicates,"vocabs_0615/predicates.txt")
# write_vocabs(eng_predicates,"vocabs_0615/eng_predicates.txt")


# Surface Tagging
write_tagging_dataset_v4_8classes(dev_grouped_t,dev_wid_list,LENGTH_FILE,"../preprocessed/surface_tagging/dev")
write_tagging_dataset_v4_8classes(train_grouped_t,train_wid_list,LENGTH_FILE,"../preprocessed/surface_tagging/train")
# There are some length errors in the original camr files (missing '0' tag in some lines), we have mannually fixed them in the processed files.
# The .sent files are collected using the sentences in camr files.



# Normalization Tagging
write_tagging_dataset_concept_norm(dev_grouped_t,dev_wid_list,LENGTH_FILE,"../preprocessed/normalization_tagging/dev")
write_tagging_dataset_concept_norm(train_grouped_t,train_wid_list,LENGTH_FILE,"../preprocessed/normalization_tagging/train")
# There are some length errors in the original camr files (missing '0' tag in some lines), we have mannually fixed them in the processed files.
# The .sent files are collected using the sentences in camr files.


# Non-aligned Tagging
# We have provided the tags under the folder "../preprocessed/non_aligned_tagging". You can run the following code after downloading the original data.
# We added 'preprocessed/non_aligned_concept_tagging/train.tag.extra_nodes_dict.clean' for Non-aligned tagging, the difference is trivial with the original extra_nodes_dict.
clean_en_dict = load_dict('../preprocessed/non_aligned_concept_tagging/train.tag.extra_nodes_dict')
train_tuples = load_tuples('../datasets/camr_tuples/tuples_train.txt')
construct_tag_files('../datasets/camr_tuples/temp/train.sent', \
    clean_en_dict, train_tuples, '../preprocessed/non_aligned_concept_tagging/train.extra_nodes.tag')


# construct validation set
dev_tuples = load_tuples('../datasets/camr_tuples/tuples_dev.txt')
construct_tag_files('../datasets/camr_tuples/temp/dev.sent', \
    clean_en_dict, dev_tuples, '../preprocessed/non_aligned_concept_tagging/dev.extra_nodes.tag')

# Relation Classification
write_4_level_relation_classification_dataset(dev_grouped_t,dev_wid_list,"../datasets/vocabs/relations.txt",LENGTH_FILE,"../preprocessed/relation_classification/dev")
write_4_level_relation_classification_dataset(train_grouped_t,train_wid_list,"../datasets/vocabs/relations.txt",LENGTH_FILE,"../preprocessed/relation_classification/train")


# Relation Alignment Classification
write_ralign_classification_dataset(dev_grouped_t,dev_wid_list,"../datasets/vocabs/relations.txt",LENGTH_FILE,"../preprocessed/relation_classification/relation_alignment_classification/dev")
write_ralign_classification_dataset(train_grouped_t,train_wid_list,"../datasets/vocabs/relations.txt",LENGTH_FILE,"../preprocessed/relation_classification/relation_alignment_classification/train")


