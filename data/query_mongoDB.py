import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from creds import client
import json
from bson import json_util
from jsonlines import jsonlines
import re

def flatten(matrix):
    return [item for row in matrix for item in row]

def parse_json(data):
    return json.loads(json_util.dumps(data))

def write_author_as_jsonl(fname, dat):
    if isinstance(dat, list):
        with open(fname, "w") as f:
            for a in dat:
                json.dump(parse_json(a), f)
                f.write("\n")

def read_jsonl(fname):
    out = []
    with jsonlines.open(fname) as reader:
        for obj in reader:
            out.append(obj)
    return out

def write_data():
    sorted_fnames = sorted(list(RAW_DAT_DIR.glob("*jsonl")))
    for author in sorted_fnames:
        nverts, simps = [], []
        author_dat = read_jsonl(author)
        for paper in author_dat:
            paper_authors = [str(a['authorId'])  for a in paper['authors']]
            # we remove None. Rare enough it shouldn't influence anything.
            paper_authors = [a for a in paper_authors if str(a) != "None"]        
            nverts.append(str(len(paper_authors)))
            simps.append(str(",".join(paper_authors)))
        
        with open("coauth-UVM-seqs.txt", "a") as f:
            f.write(",".join(nverts) + ";" + ",".join(simps)+"\n")

def read_data():
    with open("coauth-UVM-seqs.txt") as f:
        set_seq = []
        for i, line in enumerate(f.readlines()):
            if len(line.split(";")) == 2:
                nverts, simps = line.split(";")
                nverts = [int(v) for v in nverts.split(",")]
                simps = [int(v) for v in simps.split(",")]
                user_set_seq = []
                curr_ind = 0
                for nv in nverts:
                    simp = set(simps[curr_ind:(curr_ind+nv)])
                    curr_ind += nv
                    user_set_seq.append(simp)
                set_seq.append(user_set_seq)
            else:
                print(f"We skipped {line} (line {i})")
    return set_seq

ROOT_DIR = Path("..")
DAT_DIR = ROOT_DIR / "data"
RAW_DAT_DIR = DAT_DIR / "raw"

db = client['papersDB']

def prep_data():
    with open("core_faculty.txt") as f:
        name2lab = {}
        for line in f.readlines():
            name, authorId, paper = line.split(", ")
            short_name = re.sub("(\.)", "", name)
            short_name = re.sub("( |\'|-)", "_", short_name.lower())
            author_dat = list(db.papers.find({'authors.authorId': authorId}))
            write_author_as_jsonl(f"raw/{authorId}.jsonl", author_dat)
            name2lab.update({authorId: name})
    return name2lab

name2lab = prep_data()

def create_nodes():
    out = []
    with open("core_faculty.txt") as f:
        for line in f.readlines():
            _, authorId, _ = line.split(", ")
            out.append({'id': authorId, 'label': name2lab[authorId], 'group': 1})
    return out


nodes = create_nodes()
faculty_authorIDs = [n['id'] for n in nodes]

def create_edges():
    sorted_fnames = sorted(list(RAW_DAT_DIR.glob("*jsonl")))
    links = []
    for author in sorted_fnames:
        sourceId = re.sub("\.jsonl", "", str(author).split("/")[-1])
        author_dat = read_jsonl(author)
        print(f"{sourceId} has {len(author_dat)} papers")
        dict_collab = {}
        for paper in author_dat:
            relevant_collab = [author['authorId'] for author in paper['authors'] 
                               if author['authorId'] in faculty_authorIDs]
            for collabId in relevant_collab:
                if dict_collab.get(collabId) is None:
                    dict_collab[collabId] = 1
                else:
                    dict_collab[collabId] += 1

        links += [{'source': sourceId, 'target': k, 'value': v} for k,v in dict_collab.items()]
    return links

links = create_edges()

with open("csys_collab.json", "w") as f:
    json.dump({'nodes': nodes, 'links': links}, f)

# write_data()


