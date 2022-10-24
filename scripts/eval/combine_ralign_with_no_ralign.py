function_words=[]
with open("../../datasets/vocabs/ralign.txt","r") as f:
    for i in f.readlines():
        function_words.append(i.split(" ")[0])

def combine(ralign_path,no_align_path,node_path):
    align_relations=[]
    noalign_relations=[]
    nodes=[]
    ids=[]
    out_path=no_align_path+".with_func_words"

    print(no_align_path)

    with open(no_align_path,"r") as f:
        for i in f.readlines():
            noalign_relations.append(eval(i.strip().split("\t")[1]))
            ids.append(i.strip().split("\t")[0])
    
    with open(ralign_path,"r") as f:
        for i in f.readlines():
            align_relations.append(eval(i.strip().split("\t")[1]))
    
    with open(node_path,"r") as f:
        for i in f.readlines():
            nodes.append(eval(i.strip()))
    
    new_relations=[]
    print(len(align_relations))
    assert len(align_relations) == len(noalign_relations) == len(nodes)

    for ralign,no_ralign,nods in zip(align_relations,noalign_relations,nodes):
        nod_dict =  {i[0]:i[1] for i in nods}
        cur_relation=[]
        for rel in no_ralign:
            keep=True
            if rel in ralign:
                # has functional words
                for r_rel in ralign:
                    if rel[0]==r_rel[0] and rel[1]==r_rel[1] and rel[2]!=r_rel[2] and nod_dict[r_rel[2]] in function_words:
                        cur_relation.append([rel[0],rel[1],r_rel[2],nod_dict[r_rel[2]],rel[2]])
                        keep=False
                        continue
            if keep:
                cur_relation.append(rel)
        
        new_relations.append(cur_relation)

    with open(out_path,"w") as f:
        for i,ids in zip(new_relations,ids):
            f.write(ids+"\t"+str(i)+"\n")


def combine_v2(ralign_path,no_align_path,node_path):
    # higher recall, add all relations from ralign to no_align
    align_relations=[]
    noalign_relations=[]
    nodes=[]
    ids=[]
    out_path=no_align_path+".with_func_words_v2"

    print(no_align_path)

    with open(no_align_path,"r") as f:
        for i in f.readlines():
            noalign_relations.append(eval(i.strip().split("\t")[1]))
            ids.append(i.strip().split("\t")[0])
    
    with open(ralign_path,"r") as f:
        for i in f.readlines():
            align_relations.append(eval(i.strip().split("\t")[1]))
    
    with open(node_path,"r") as f:
        for i in f.readlines():
            nodes.append(eval(i.strip()))
    
    new_relations=[]
    print(len(align_relations))
    assert len(align_relations) == len(noalign_relations) == len(nodes)

    for ralign,no_ralign,nods in zip(align_relations,noalign_relations,nodes):
        nod_dict =  {i[0]:i[1] for i in nods}
        cur_relation=[]
        cur_align_relations=[]
        for r_rel in ralign:
            for r_rel_2 in ralign:
                if r_rel[0]==r_rel_2[0] and r_rel[1]==r_rel_2[1] and r_rel[2]!=r_rel_2[2] and nod_dict[r_rel_2[2]] in function_words:
                    cur_align_relations.append([r_rel[0],r_rel[1],r_rel_2[2],nod_dict[r_rel_2[2]],r_rel[2]])
                    continue
        
        cur_relation = cur_relation+cur_align_relations

        for rel in no_ralign:
            keep=True
            for align_rel in cur_align_relations:
                if rel[0]==align_rel[0] and rel[1]==align_rel[1] and rel[2] == align_rel[4]:
                    keep=False
                    continue
            if keep:
                cur_relation.append(rel)
        
        new_relations.append(cur_relation)

    with open(out_path,"w") as f:
        for i,ids in zip(new_relations,ids):
            f.write(ids+"\t"+str(i)+"\n")




import sys
combine(sys.argv[1],sys.argv[2],sys.argv[3])
