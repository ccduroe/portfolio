# Pancreas Cell Type Classification

## Project Overview
Machine learning project to classify pancreas cell types using machine learning techniques, developed as part of an Applied Machine Learning course.

## Inspiration and Methodology
This project was inspired by the research paper:

**CTISL: A Dynamic Stacking Multi-Class Classification Approach for Identifying Cell Types from Single-Cell RNA-Sequencing Data**

- **Original Paper**: [CTICL Paper on Oxford Academic](https://doi.org/10.1093/bioinformatics/btae063)
- **Dataset Source**: [Zenodo Record](https://zenodo.org/records/10568906)

While building upon the foundational work of the original researchers, this project implements custom machine learning approaches, alternative classification techniques, and comparative analysis of different models.

## Key Features
- Multi-model cell type classification
- Cross-species dataset (Mouse and Human Pancreas)
- Machine learning techniques
  - Support Vector Machines
  - Decision Trees
  - Boosted Trees
  - Random Forest
  - Neural Networks
- Dimensionality Reduction using PCA
- Imbalanced Data Handling with SMOTE

## Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scanpy](https://img.shields.io/badge/Scanpy-049A9B?style=for-the-badge)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-D3756B?style=for-the-badge)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ccduroe/portfolio.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd portfolio/projects/pancreas_classification
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
To run the main analysis script, execute the following command from the project root directory:
```bash
python main.py
```

## Project Structure

portfolio/projects/pancreas_classification/
├── src/                # Source code modules
├── data/            # Dataset files
├── docs/            # Project documentation
└── main.py             # Main script to run the project

## Key Achievements
- Developed and compared  machine learning models for cell type classification
- Implemented advanced preprocessing techniques for single-cell RNA-seq data
- Successfully handled challenges associated with cross-species dataset

## Future Work
- Explore more advanced feature engineering and selection methods
- Implement additional deep learning architectures (e.g., Transformers)
- Expand the analysis to include more diverse datasets

## Citation
If you use insights or code from this project, please reference:

- Original Paper: [CTISL Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10873586/#sup1)
- This Repository: [Pancreas Cell Type Classification on GitHub](https://github.com/ccduroe/portfolio/tree/main/projects/pancreas_classification)

## Contact
- **GitHub**: [ccduroe](https://github.com/ccduroe)
- **Email**: [ccduroe@gmail.com](mailto:ccduroe@gmail.com)

## License
This project is licensed under the MIT license - see the [`LICENSE`](LICENSE) file for details. 
