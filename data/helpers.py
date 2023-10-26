from pathlib import Path
import numpy as np
from jsonlines import jsonlines
from itertools import combinations

data_dir = Path()
RAW_DAT_DIR = data_dir / 'raw'


def is_valid_paper(paper):
    if not isinstance(paper, dict):
        return False
    if 'authors' not in paper or 'year' not in paper:
        return False
    if not isinstance(paper['authors'], list) or not isinstance(paper['year'], int):
        return False
    return True

def get_author_ids(paper):
    return [author['authorId'] for author in paper['authors'] if 'authorId' in author and author['authorId'] is not None]

def analyze_collaborations(papers, thresh_nb_auth=100):
    if not isinstance(papers, list) or len(papers) < 1:
        print("Input should be a list of papers with at least one paper.")
        return {}
    
    dict_collab = {}
    name2lab = {}
    for paper in papers:
        if not is_valid_paper(paper):
            print("Invalid paper format. Skipping:", paper)
            continue
        
        author_ids = get_author_ids(paper)

        if len(author_ids) < thresh_nb_auth:

            name2lab.update({author['authorId']:author['name'] for author in paper['authors'] if author['authorId'] is not None})
            
            if not author_ids:
                print("No valid authors found in paper. Skipping:", paper)
                continue
            
            year = paper['year']
            for i in range(len(author_ids)):
                for j in range(i + 1, len(author_ids)):
                    key = ((author_ids[i], author_ids[j]), year) if author_ids[i] < author_ids[j] else ((author_ids[j], author_ids[i]), year)
                    dict_collab[key] = dict_collab.get(key, 0) + 1
    
    return dict_collab, name2lab

papers = [
    {
        'title': 'Paper 1',
        'authors': [{'authorId': '1', 'name': 'Author A'}, {'authorId': '2', 'name': 'Author B'}],
        'year': 2022
    },
    {
        'title': 'Paper 2',
        'authors': [{'authorId': '2', 'name': 'Author B'}, {'authorId': '3', 'name': 'Author C'}],
        'year': 2022
    },
    {
        'title': 'Paper 3',
        'authors': [{'authorId': '1', 'name': 'Author A'}, {'authorId': '3', 'name': 'Author C'}],
        'year': 2022
    },
    {
        'title': 'Paper 3',
        'authors': [{'authorId': '2', 'name': 'Author B'}, {'authorId': '1', 'name': 'Author A'}],
        'year': 2022
    },
]

result, lab = analyze_collaborations(papers)

# Check when tuples are out of order, we still catch that in the same tuple
assert result[(('1', '2'), 2022)] == 2


