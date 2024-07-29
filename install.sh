#!/bin/sh

pip install scipy==1.13.1
pip install matplotlib==3.9.1
pip install torch==2.1.2
pip install tqdm==4.66.4
pip install python-Levenshtein==0.25.1
pip install datasets==2.14.6
pip install pycocoevalcap==1.2
pip install git+https://github.com/openai/CLIP.git
pip install scikit-learn==1.5.1
pip install spacy==3.7.5
pip install salesforce-lavis==1.0.2
pip install polos==0.1.3
pip install sentence-transformers==3.0.1
pip install diffusers==0.29.2

# Prepare clipscore
git clone https://github.com/jmhessel/clipscore.git

# Prepare PACScore
git clone https://github.com/aimagelab/pacscore.git

# Prepare content overlap metrics
git clone https://github.com/DavidMChan/caption-by-committee.git
mv caption-by-committee/cbc/metrics/content_score.py .
rm -rf caption-by-committee
python -m spacy download en_core_web_sm
