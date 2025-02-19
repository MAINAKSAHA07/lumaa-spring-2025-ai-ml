# Movie Recommendation System

## Overview

This project is a simple content-based movie recommendation system that suggests movies based on user input. The system leverages TF-IDF vectorization and cosine similarity to find movies that closely match user preferences.

## Features

- **Content-based filtering**: Uses movie descriptions, cast, and crew data for recommendations.
- **TF-IDF Vectorization**: Converts text descriptions into numerical representations.
- **Cosine Similarity**: Measures the similarity between user input and movie descriptions.
- **JSON Parsing**: Extracts cast and crew details from structured data.
- **Command-Line Interface (CLI)**: Allows users to enter queries and receive recommendations instantly.

## Technologies Used

- **Python 3.8+**
- **Pandas** (for data handling)
- **NumPy** (for numerical operations)
- **Scikit-learn** (for TF-IDF and similarity calculations)
- **JSON** (for parsing movie metadata)

## Setup Instructions

### 1. Clone the Repository
```bash
 git clone https://github.com/MAINAKSAHA07/lumaa-spring-2025-ai-ml.git
 cd lumaa-spring-2025-ai-ml
```

### 2. Create a Virtual Environment
```bash
 python -m venv venv
 source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
 pip install -r requirements.txt
```

### 4. Prepare the Dataset
Ensure that your dataset file `movies_data.csv` is present in the project directory. The dataset should contain the following columns:

- `title`: Movie title
- `cast`: JSON-formatted list of cast members
- `crew`: JSON-formatted list of crew members

Example format:
```json
[{"name": "Leonardo DiCaprio"}, {"name": "Kate Winslet"}]
```

### 5. Run the Recommendation System
```bash
 python content_recommender.py
```

## Usage Guide

1. The program will prompt you to enter your movie preferences. Example queries:
   - "Sci-fi movies with space exploration"
   - "Comedy movies with famous stand-up comedians"
   - "Thriller movies directed by Christopher Nolan"

2. The system will analyze your input and return the top 5 most relevant movies.

### Example Output:
```
Enter your movie preferences: Action movies with superheroes

Recommended Movies:
--------------------------------------------------
1. The Dark Knight
   Similarity Score: 0.4152
   Cast: Christian Bale, Heath Ledger, Aaron Eckhart
--------------------------------------------------
2. Avengers: Endgame
   Similarity Score: 0.3876
   Cast: Robert Downey Jr., Chris Evans, Mark Ruffalo
--------------------------------------------------
[...]
```

## How It Works

1. **Load Dataset**: Reads movie data from `movies_data.csv`.
2. **Preprocess Data**: Extracts cast and crew names from JSON format.
3. **TF-IDF Vectorization**: Converts text descriptions into feature vectors.
4. **Similarity Calculation**: Uses cosine similarity to find closest matches.
5. **Recommendation Generation**: Returns the top N recommended movies.

## Future Enhancements

- Add **genre-based filtering** to refine recommendations.
- Implement **user-based collaborative filtering**.
- Enhance the UI with a **web-based interface** using Flask or FastAPI.

## Contribution Guidelines

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.

## Salary Expectation

- **$25/hr** for freelance development or contractual work.

## Demo
A demo video showcasing the recommendation system can be found here: 

https://github.com/user-attachments/assets/5cef92ea-0f86-4222-b5b3-f8ee48f135d7


---
### Author: [Mainak Saha](https://github.com/MAINAKSAHA07)
GitHub Repository: [lumaa-spring-2025-ai-ml](https://github.com/MAINAKSAHA07/lumaa-spring-2025-ai-ml)

