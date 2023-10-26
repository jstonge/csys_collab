from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from jsonlines import jsonlines
import re
import graph_tool.all as gt
import seaborn as sns
from collections import Counter
import pandas as pd
from helpers import analyze_collaborations


# plt.switch_backend("cairo")
sns.set_style("white")

data_dir = Path()
RAW_DAT_DIR = data_dir / 'raw'

core_faculty = pd.read_csv("core_faculty.txt", names=['name', 'authorId', 'title'])

def count_paper_by_faculty():
  for fname in RAW_DAT_DIR.glob("*jsonl"):
    papers = []
    with jsonlines.open(fname) as reader:
      for paper in reader:
        papers.append(paper)

    print((fname, len(papers)))

def get_papers():
  papers = []
  for fname in RAW_DAT_DIR.glob("*jsonl"):
    with jsonlines.open(fname) as reader:
      for paper in reader:
        papers.append(paper)
  return papers

def create_edgelist(collab_dict, years=[2015]):
  links = []
  for e,w in list(collab_dict.items()):
    source, target = e[0]
    year = e[1]
    if year in years:
      links.append((source, target, year, w))
  return links

def create_graph():
  g = gt.Graph(edgelist, hashed=True, 
               eprops=[('year', 'int'), ('weigth', 'int')], 
               directed=False)

  g.vp["label"] = g.new_vp('string')
  g.vp["is_core"] = g.new_vp('bool')
  for v in g.vertices():
    g.vp['label'][v] = lab[g.vp['ids'][v]]
    g.vp['is_core'][v] = True if int(g.vp['ids'][v]) in core_faculty['authorId'].tolist() else False

  # check graph size
  print((g.num_edges(), g.num_vertices()))
  return g

papers = get_papers()

# paper with too many authors make eveyrthing weird
# paper1993 = [p for p in papers if p.get('year') and p['year'] == 1993]

collab_dict, lab = analyze_collaborations(papers, thresh_nb_auth=20)
import csv
# Q: whatsup with 1993?
# A: there were 4 papers but one with 55 coauthors...we put a threshold
[(y, f"#papers: {len(create_edgelist(collab_dict, years=[y]))}") for y in range(1990, 2023)]


for yr in range(2000, 2023):
  edgelist = create_edgelist(collab_dict, years=[yr])
 
  with open(f"csys_graph/edgelist_{yr}.txt", 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerows(edgelist)
    

  g = create_graph()

  np.savetxt(f"csys_graph/collab_{yr}.txt",gt.adjacency(g).todense())

  # check vertex properties
  # [(g.vp['ids'][v], g.vp['label'][v], g.vp['is_core'][v]) for v in g.vertices()]

  # plotting
  # no network properties
  # gt.graph_draw(g, pos=gt.sfdp_layout(g),
  #               vertex_color="black", 
  #               vertex_fill_color=g.vp['is_core'], 
  #               bg_color="white")

  # using network properties
  pos = gt.sfdp_layout(g)
  deg = g.degree_property_map("total")
  deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
  vbet, ebet = gt.betweenness(g)
  ebet.a /= ebet.a.max() / 10.

  fig, ax = plt.subplots(figsize=(20,20));
  gt.graph_draw(g, pos=pos,
                vertex_color=vbet, 
                vertex_fill_color=g.vp['is_core'], 
                vertex_size=gt.prop_to_size(vbet, 0.1, 0.5), 
                vertex_pen_width=0.1,
                vertex_text=g.vp['label'],
                vertex_font_size=gt.prop_to_size(vbet, 0.1, 0.5),
                edge_pen_width=gt.prop_to_size(g.ep['weigth'],0.01,0.1),
                bg_color="white",
                mplfig=ax);
  ax.set_title(f"Graph of coauthorship in {yr} (node size propto betweenness; width propto #coauthored papers; node pen width color prop to betweenness)");
  fig.savefig(f"graph-bet/graph-draw-betweenness{yr}.pdf")



# Graph filtering
# g2015 = gt.GraphView(g, efilt=g.ep['year'].a == 2015)

# lower than biology in 2000s, but at the same it might not be suprising
# given how we constructed the network...
assort, cc_assort = gt.assortativity(g, deg='total')

# clustering
clust = gt.global_clustering(g, weight=g.ep['weigth'])

# degree distirbution
counts, bins = gt.vertex_hist(g, deg='total')
sns.histplot(counts, bins=bins)
plt.xlabel("# coauthors")
plt.ylabel("frequency")
sns.despine()
plt.tight_layout()

# largest component
u = gt.extract_largest_component(g)

(u.num_edges(), u.num_vertices())

gt.graph_draw(u)


###### LOOKING AT HOW AVERAGE NUMBER OF COAUTHORS PER YEAR CHANGES OVER TIME ####

dfs = []
for i, row in core_faculty.iterrows():
  years = []
  nb_coauth_by_yrs = []
  nb_papers = []
  for year in range(2000, 2023):
    papers_by_faculty = []
    for p in papers:
      if p.get('year') and p['year'] == year:
        authorIds = [int(a['authorId']) for a in p['authors'] if a['authorId'] is not None]
        if authorIds is not None:
          if row.authorId in authorIds:
            papers_by_faculty.append(len(authorIds))
    years.append(year)
    nb_coauth_by_yrs.append(np.mean(papers_by_faculty))
    nb_papers.append(len(papers_by_faculty))
       
  dfs.append(pd.DataFrame({
    'year': years, 
    'avg_nb_coauthor': nb_coauth_by_yrs,
    'nb_papers': nb_papers,
    'authorId': row.authorId,
    'name': row['name']
    }))

df = pd.concat(dfs,axis=0)
df['authorId'] = df.authorId.astype(str)

paper_count = df.groupby("authorId")["nb_papers"].sum().reset_index(name='n')
people2remove = paper_count.query("n < 30").authorId.tolist()

sub_df = df.loc[~df.authorId.isin(people2remove)].reset_index(drop=True)
df_mean = sub_df.groupby("year")['avg_nb_coauthor'].mean().reset_index()

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(x='year', y='avg_nb_coauthor', hue='name', data=sub_df,
             alpha=0.4, markers=True, style="name", dashes=False, ax=ax);
sns.lineplot(x='year', y='avg_nb_coauthor', color="red", 
             linestyle="dashed", data=df_mean, ax=ax)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1));
sns.despine()
plt.show()
