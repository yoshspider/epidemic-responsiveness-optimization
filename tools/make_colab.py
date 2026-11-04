"""Build portable notebook and small source bundle without raw datasets/dependencies."""
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/"colab"
out.mkdir(exist_ok=True)
cells=[]
def markdown(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
def code(text):
    cells.append({"cell_type":"code","metadata":{},"source":text.splitlines(True),"outputs":[],"execution_count":None})
markdown("# Epidemic allocation research\nUpload research_source.zip from this project's colab folder. Select a GPU runtime to use optional PyTorch CUDA lookahead; auto backend falls back to NumPy on CPU. Run parity tests before experiments. Synthetic validation is explicitly labeled. Prepare empirical data separately and record provenance. Colab execution and account login are performed by you.")
code("from google.colab import files\nfrom pathlib import Path\nimport zipfile\nfiles.upload()\nwith zipfile.ZipFile('research_source.zip') as bundle:\n    bundle.extractall('/content/epidemic_research')\n%cd /content/epidemic_research\n%pip install -q -r requirements.txt")
code("!python -m unittest discover -s tests -v\n!python -m epidemic.benchmark --counties 3126 --horizon 28 --backend auto\nimport json\nconfig=json.load(open('config.example.json'))\nconfig['rollout_backend']='auto'\nconfig['stochastic']=True\njson.dump(config,open('colab_config.json','w'),indent=2)")
markdown("## Synthetic debugging example\nThese outcomes are not empirical evidence. Use several seeds only with stochastic execution; deterministic seeds repeat the same outcome.")
code("!python -m epidemic.experiments --synthetic --config colab_config.json --counties 24 --days 28 --seeds 3 --output outputs/demo\nfrom IPython.display import Image, display\ndisplay(Image('outputs/demo/infections.png'))")
markdown("## Empirical inputs\nUpload a prepared_data.zip with counties.csv and mobility.csv at its root, after following README.md. CTIS training targets and TBCOV features must be temporally aligned. The CTIS mask signal starts September 8, 2020, so July 2020 decisions using CTIS training are retrospective. Intervention cost assumptions must be explicit.")
code("# Uncomment after preparing and uploading the empirical bundle.\n# files.upload()\n# with zipfile.ZipFile('prepared_data.zip') as bundle:\n#     bundle.extractall('data/prepared')\n# !python -m epidemic.experiments --data data/prepared --config config.example.json --days 84 --seeds 20 --output outputs/empirical")
markdown("## TBCOV streaming\nUse sentiment_label and resident-location fields from base TSV releases; hydration is unnecessary for these labels. Monthly archives are large. Download only selected months to sufficiently large storage, then aggregate in chunks with epidemic.tbcov. Country/language ID lists alone do not contain the base labels. Do not treat place mentions in tweet text as residence.")
code("# !python -m epidemic.tbcov selected_month.tsv county_crosswalk.csv --output outputs/tbcov_county_month.csv\n# !python -m epidemic.prediction features.csv feature_sets.json --folds 5\n# !python -m epidemic.sensitivity --synthetic --days 28 --seeds 5")
code("import shutil\nshutil.make_archive('/content/research_outputs','zip','outputs')\nfiles.download('/content/research_outputs.zip')")
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
(out/"epidemic_research.ipynb").write_text(json.dumps(notebook,indent=2))
with zipfile.ZipFile(out/"research_source.zip","w",zipfile.ZIP_DEFLATED) as bundle:
    for folder in ("epidemic","tests"):
        for file in (root/folder).glob("*.py"):
            bundle.write(file,file.relative_to(root))
    for name in ("requirements.txt","README.md","DATA_SOURCES.md","config.example.json"):
        bundle.write(root/name,name)
print(out)
