Temporal link prediction with graph neural networks
==============================

To replicate the work one should run the following notebooks:

- generate_data.ipynb
- predict_static_graphs.ipynb
- networkit.ipynb (This was run in Google Colab, because networkit is not supported on Windows)
- analyze_data.ipynb

Main requirements: 

- Python 3.7.9
- torch==1.7.1
- numpy==1.21.2
- pandas==1.1.0
- scikit-learn==0.24.1
- scipy==1.7.1
- networkit was used in a Google Colab environment due to its OS constraints

The whole pip list and conda env can be found in the corresponding files.
