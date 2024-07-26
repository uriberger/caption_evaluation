#!/bin/sh

# Prepare clipscore
git clone https://github.com/jmhessel/clipscore.git

# Prepare PACScore
git clone https://github.com/aimagelab/pacscore.git

# Prepare content overlap metrics
git clone https://github.com/DavidMChan/caption-by-committee.git
mv caption-by-committee/cbc/metrics/content_score.py .
rm -rf caption-by-committee
python -m spacy download en_core_web_sm
