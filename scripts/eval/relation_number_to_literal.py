function_words=[]
with open("../../datasets/vocabs/ralign.txt","r") as f:
    for i in f.readlines():
        function_words.append(i.split(" ")[0])


def labelsfile_to_relation(nodes_path,relation_path,relation_dict_path,out_path,id_copy_path="/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/dev.4level.relations.changed.literal"):
    relation_index={}
    with open(relation_dict_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            relation_index[i+1]=j.strip().split()[0]
    
    with open(nodes_path,"r") as f:
        nodes = f.readlines()
    
    with open(relation_path,"r") as f:
        relations = f.readlines()
    
    literal_relations_list=[]
    
    for nod,rel in zip(nodes,relations):
        current_literal_relation=[]
        cur_nodes = [i[0] for i in eval(nod)]
        cur_relation = eval(rel)

        nod_dict =  {i[0]:i[1] for i in eval(nod)}

        print(nod)


        for i in range(len(cur_relation)):
            temp_4={}
            for j in range(len(cur_relation)):
                if cur_relation[i][j] != 0:
                    relation = relation_index[cur_relation[i][j]]
                    node1 = cur_nodes[i].replace("b","x")
                    node2 = cur_nodes[j].replace("b","x")
                    
                    if relation in temp_4.keys() and len(temp_4[relation])==1:
                        temp_4[relation].append(node2)
                        # maybe align relation

                        print(i,temp_4,relation,node1,node2)

                        poss_align_1,poss_align_2 = temp_4[relation][0],temp_4[relation][1]

                        if nod_dict[poss_align_1] in function_words:
                            current_literal_relation.remove([node1,relation,poss_align_1])
                            current_literal_relation.append([node1,relation,poss_align_1,nod_dict[poss_align_1],poss_align_2])
                        elif nod_dict[poss_align_2] in function_words:
                            current_literal_relation.remove([node1,relation,poss_align_1])
                            current_literal_relation.append([node1,relation,poss_align_2,nod_dict[poss_align_2],poss_align_1])
                        else:
                            current_literal_relation.append([node1,relation,node2])

                    else:
                        temp_4[relation] = [node2]
                        current_literal_relation.append([node1,relation,node2])
            
        


        literal_relations_list.append(current_literal_relation)
    
    with open(id_copy_path,"r") as f:
        ids = [i.split("\t")[0] for i in f.readlines()]

    with open(out_path,"w") as f:
        for i,ids in zip(literal_relations_list,ids):
            f.write(ids+"\t"+str(i)+"\n")

def labelsfile_to_relation_old(nodes_path,relation_path,relation_dict_path,out_path,id_copy_path="/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/dev.4level.relations.changed.literal"):
    relation_index={}
    with open(relation_dict_path,"r") as f:
        lines = f.readlines()
        for i,j in enumerate(lines):
            relation_index[i+1]=j.strip().split()[0]
    
    with open(nodes_path,"r") as f:
        nodes = f.readlines()
    
    with open(relation_path,"r") as f:
        relations = f.readlines()
    
    literal_relations_list=[]
    
    for nod,rel in zip(nodes,relations):
        current_literal_relation=[]
        cur_nodes = [i[0] for i in eval(nod)]
        cur_relation = eval(rel)

        nod_dict =  {i[0]:i[1] for i in eval(nod)}

        print(nod)


        for i in range(len(cur_relation)):
            temp_4={}
            for j in range(len(cur_relation)):
                if cur_relation[i][j] != 0:
                    
                    if cur_relation[i][j] not in relation_index.keys():
                        continue

                    relation = relation_index[cur_relation[i][j]]
                    node1 = cur_nodes[i].replace("b","x")
                    node2 = cur_nodes[j].replace("b","x")

                    # arg0-of 和 arg0 只保留一个
                    # if cur_relation[j][i] != 0 and "-of" not in relation:
                    #     continue
                    
                    current_literal_relation.append([node1,relation,node2])
            
        


        literal_relations_list.append(current_literal_relation)
    
    with open(id_copy_path,"r") as f:
        ids = [i.split("\t")[0] for i in f.readlines()]

    with open(out_path,"w") as f:
        for i,ids in zip(literal_relations_list,ids):
            f.write(ids+"\t"+str(i)+"\n")

import sys

print(sys.argv)

NODE_FILE=sys.argv[1]
RELATION_FILE=sys.argv[2]
OUTPUT=sys.argv[3]
VOCAB_FILE="../../datasets/vocabs/relations.txt"


id_copy_path="../../test_A/test_A_with_id.txt"

labelsfile_to_relation_old(NODE_FILE,RELATION_FILE,VOCAB_FILE,OUTPUT,id_copy_path)
