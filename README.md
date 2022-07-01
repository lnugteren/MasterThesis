### Ranking semi-structured imperfect data based on novelty and relevance using text summarisation models
#### An evaluation of the generalisability and transferability of results across domains.
---

Welcome to this Github for my Master Thesis in Data Science (with the same title). 

Here, you will find 
- The implementations of all the models used (for each experiment),
- The raw results for all the experiments,
- The evaluation codes and raw evaluations.

It it ordered by research question, test, category, and number of reviews (`nr_rev`).

---

### Example run
___

If you want to run the code yourself, you should first download the data you want to use [here](http://deepyeti.ucsd.edu/jianmo/amazon/). 

After downloading the data, store it in a folder called 'Data'. Check whether the files are still zipped and consider unzipping them. Large files take a long time, it is recommended to use `model_RQ1.py` to first separate large jsons into smaller csvs. Put all these csvs in a folder within 'Data' called 'csvs_{category}' or, if meta data, 'csvs_meta_{category}'.

If all goes well, you can either slightly adjust the `.py` file to loop through the different categories and `nr_revs` themselves or you can use the terminal. 

After opening your terminal in the same folder as your 'Data' and codes location an example run in the terminal would be:

**RQ1:** `python3 model_RQ2.py {category} {nr_rev}` for instance `python3 model_RQ1.py Office_Products 50`

**RQ2:** `python3 model_RQ1.py {test} {category}` for instance `python3 model_RQ1.py test3 Home_and_Kitchen`

**RQ3b:** `python3 model_RQ2.py {time-window} {category}` for instance `python3 model_RQ1.py 90 Automotive`

> Note: The loading in can take quite a while so do not be alarmed.
