# Assignment-Topsis_PretrainedModel

## Overview

This assignment/project aims to evaluate pretrained text summarization models using the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method. The project also incorporates additional evaluation metrics such as BERTScore, METEOR, ROUGE, and BLEU to assess the performance of popular pretrained models.

## Models Evaluated

The following pretrained summarization models have been considered for evaluation:

1. BERT (Bidirectional Encoder Representations from Transformers)
2. T5 (Text-To-Text Transfer Transformer)
3. XLNet
4. RoBERTa (Robustly optimized BERT approach)
5. PEGASUS

## Evaluation Metrics

### TOPSIS

The Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) is a decision-making method that evaluates the performance of alternative solutions. It considers multiple criteria and ranks solutions based on their proximity to the ideal solution and remoteness from the negative ideal solution.

### BERTScore

BERTScore measures the similarity between the contextual embeddings of model-generated text and human reference text using pre-trained BERT models. The score ranges from 0 to 1, with higher scores indicating better similarity to human references.

### METEOR

METEOR (Metric for Evaluation of Translation with Explicit ORdering) considers precision, recall, stemming, and synonymy to evaluate the quality of machine-generated text. The score ranges from 0 to 1, with higher scores indicating better alignment with human references.

### ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of summaries by comparing them to human reference summaries. Common ROUGE metrics include ROUGE-N (n-gram overlap) and ROUGE-L (longest common subsequence). Scores vary depending on the specific metric but generally range from 0 to 1.

### BLEU (Bilingual Evaluation Understudy)

BLEU measures the precision of a generated summary by comparing it to one or more human reference summaries. It considers n-gram overlap between the generated and reference text. BLEU scores range from 0 to 1, with higher scores indicating better agreement with human references.

## Instructions

1. Install the required libraries by running `pip install -r requirements.txt`.
2. Run the evaluation script for each model with specific summarization tasks and datasets.
3. Explore the generated evaluation metrics, including TOPSIS, BERTScore, METEOR, ROUGE, and BLEU.
4. Analyze and interpret the results to understand the relative performance of each pretrained summarization model.

## Author

- Cheshta Biala


## License

This project is licensed under the [License Name] License - see the [LICENSE.md](LICENSE.md) file for details.
