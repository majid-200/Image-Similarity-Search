# AI-Powered Image Similarity Search

This repository contains a collection of notebooks that build and explore an AI-powered visual search engine. The core idea is to take an image and find the most visually similar images from a large collection, similar to a "reverse image search" but for your own photo library.

The project demonstrates an end-to-end workflow: from extracting meaningful features from images using deep learning models to indexing them for high-speed retrieval using specialized libraries like FAISS.

## Key Features

-   **End-to-End System**: A complete, runnable system for indexing a directory of images and performing searches.
-   **State-of-the-Art Models**: Utilizes a pre-trained **Vision Transformer (ViT)** for high-quality feature extraction.
-   **High-Speed Search**: Implements **FAISS (Facebook AI Similarity Search)** for efficient, approximate nearest neighbor search, allowing it to scale to millions of images.
-   **Comparative Analysis**: Includes in-depth explorations of different vector search strategies (Brute-Force vs. FAISS) and feature extraction models (ViT vs. VGG).
-   **Visualization**: Uses PCA and Matplotlib to visualize the high-dimensional feature space, helping to build an intuition for how the models "see" image similarity.

---

## Notebooks in this Repository

This project is broken down into three main Jupyter notebooks, each with a specific focus.

### 1. `03_Efficient_Image_Retrieval_with_Vision_Transformer_(ViT)_and_FAISS` - The Complete Application

This notebook builds the main, end-to-end image retrieval system. It is the core application of this project.

**What it does:**
-   **Feature Extractor**: A class `ImageFeatureExtractor` is built around a pre-trained Vision Transformer (`vit_b_16`) from PyTorch. It's customized to output 768-dimensional feature vectors (embeddings) that represent the visual content of an image.
-   **FAISS Indexing**: It uses the powerful `IndexIVFFlat` from FAISS. This index first clusters all image vectors into regions (an "inverted file") and then only searches the most relevant regions for a given query, making the search process incredibly fast.
-   **System Class**: The `ImageRetrievalSystem` class ties everything together. It handles:
    -   Indexing an entire directory of images.
    -   Saving the indexed database and metadata to disk for persistence.
    -   Loading a pre-built index to perform new searches without re-indexing.
    -   Searching for similar images and returning a ranked list of results with similarity scores.
-   **Runner Function**: A main `run_image_retrieval` function provides a simple interface to either `index` a new dataset or `search` an existing one.

### 2. `02_Understanding_FAISS_for_efficient_similarity_search_of_dense_vectors` - Understanding Vector Search

This notebook is a pedagogical exploration of the "search" part of the system. It visually and empirically compares different methods for finding similar vectors in a high-dimensional space.

**What it demonstrates:**
1.  **Method 1: Brute-Force Cosine Similarity**: A manual, pure NumPy implementation to establish a baseline. It's exact but scales poorly (O(n) complexity).
2.  **Method 2: FAISS `IndexFlatL2`**: An optimized brute-force search using FAISS. It's still an exact search but significantly faster due to C++ and SIMD optimizations.
3.  **Method 3: FAISS `IndexIVF` (Approximate Search)**: The star of the show. This section explains the trade-off between speed and accuracy. It demonstrates the two-step process:
    -   **Training**: K-means clustering is used to partition the vector space into regions.
    -   **Searching**: The search is restricted to a small number of regions (`nprobe`) closest to the query vector, leading to a massive speedup.
4.  **Performance Benchmark**: The notebook concludes with a performance test that measures the search time of each method across datasets of varying sizes (from 100 to 100,000 vectors), clearly illustrating the scalability benefits of FAISS IVF.

### 3. `01_Image_Similarity_Search_with_VGG16_and_Cosine_Distance` - A Classic CNN Approach

This notebook explores a more traditional approach to feature extraction using the VGG16 Convolutional Neural Network (CNN). It serves as a great comparison to the modern Vision Transformer approach.

**Implementations:**
-   **TensorFlow/Keras Version**: Loads a pre-trained VGG16, extracts 512-dimensional feature vectors from a set of images, and saves them to an HDF5 file for efficient storage.
-   **PyTorch Version**: Replicates the same logic in PyTorch, also offering ResNet50 as an alternative model and demonstrating how to extract features from different layers of the network (convolutional vs. fully-connected).
-   **Search & Visualization**: A simple search is performed by calculating the cosine similarity between the query and all database images. The results are visualized in two ways:
    -   Displaying the top matching images.
    -   A 3D scatter plot of the feature space, reduced to 3 dimensions using **Principal Component Analysis (PCA)**, which helps visualize how similar images cluster together.

---

## Core Concepts Explained

#### 1. Feature Extraction (Embeddings)
At its heart, this system works by converting each image into a list of numbers—a **vector** or **embedding**. This vector captures the semantic content of the image. A "cat" image will have a vector that is numerically close to other "cat" images. This project uses a pre-trained Vision Transformer (ViT) to generate these vectors.

#### 2. Vector Similarity Search
Once we have vectors, the problem becomes finding the "closest" vectors in a high-dimensional space. "Closeness" is typically measured by:
-   **Euclidean Distance (L2)**: The straight-line distance between two vector points. (Lower is better).
-   **Cosine Similarity**: The cosine of the angle between two vectors. It measures orientation, not magnitude. (Higher is better).

#### 3. FAISS (Facebook AI Similarity Search)
Searching through millions of vectors one-by-one is slow. FAISS is a library designed to solve this. It specializes in **Approximate Nearest Neighbor (ANN)** search. Instead of guaranteeing the *perfect* matches, it finds *extremely close* matches with incredible speed by using clever indexing structures like the **Inverted File (IVF)**, which pre-clusters the vectors.

---

## How to Run

#### Prerequisites
Make sure you have Python 3.8+ and the required libraries installed.

```bash
pip install torch torchvision numpy faiss-cpu Pillow scikit-learn matplotlib
```

#### Step 1: Index Your Images (One-time setup)
1.  Place all the images you want to index into a single folder (e.g., `images/base/`).
2.  In the `03_Efficient_Image_Retrieval_with_Vision_Transformer_(ViT)_and_FAISS` notebook, configure the parameters in the final cell (`if __name__ == "__main__":`).
3.  Set `TASK = "index"` and update `IMAGE_DIR` to point to your folder.
4.  Run the cell. This will create two files: `image_index.faiss` and `image_metadata.json`.

#### Step 2: Search for Similar Images
1.  Once your index is built, you can perform searches anytime.
2.  Set `TASK = "search"`.
3.  Set `QUERY_IMAGE` to the path of the image you want to find matches for.
4.  Run the same cell again. The system will load the index and print the top similar images.

#### Exploring the Other Notebooks
The `02_Understanding_FAISS_for_efficient_similarity_search_of_dense_vectors` and `01_Image_Similarity_Search_with_VGG16_and_Cosine_Distance` notebooks are designed for learning and can be run cell-by-cell in a Jupyter environment to understand the underlying concepts.

---

## Acknowledgment

The development of this project was guided by an insightful tutorial series on YouTube. The tutorials provided an excellent foundation and a clear, step-by-step walkthrough of the key concepts, including feature extraction with deep learning models and high-speed indexing with FAISS.

I am grateful to the original creator "DigitalSreeni" for sharing their knowledge. I highly recommend their content to anyone interested in this topic. You can find the tutorial playlist here:
**https://youtube.com/playlist?list=PLZsOBAyNTZwbvIDz-xWBZmRvDgZNbFiJa&feature=shared**

---
