# adobe_task_1

# ENV SETUP
python3.12 -m venv venv
source venv/bin/activate

# install miner u
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"

# extract data using mineru
mineru -p /Users/mayurgd/Documents/CodingSpace/adobe_task_1/data/inputs/annual_reports/adbe-2023-annual-report.pdf -o /Users/mayurgd/Documents/CodingSpace/adobe_task_1/data/outputs/annual_reports -b pipeline

pip install sentence_transformers faiss-cpu lxml


# sample question
What is our current revenue trend and were there key risks highlighted in the last quarter?