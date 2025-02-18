import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import sys

# Download NLTK data (uncomment if needed)
# nltk.download('stopwords')
# nltk.download('punkt')

class ContentBasedRecommender:
    def __init__(self, dataset_path, text_column, title_column=None):
        """
        Initialize the recommender with a dataset.
        
        Args:
            dataset_path (str): Path to the CSV file containing the dataset
            text_column (str): Column name containing the text to use for recommendations
            title_column (str, optional): Column name containing the item title/name
        """
        self.df = pd.read_csv(dataset_path)
        self.text_column = text_column
        self.title_column = title_column if title_column else text_column
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Fill NaN values in text column with empty strings
        self.df[self.text_column] = self.df[self.text_column].fillna('')
        
        # Initialize and fit the vectorizer
        self._init_vectorizer()
    
    def _init_vectorizer(self):
        """Initialize and fit the TF-IDF vectorizer on the dataset"""
        stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df[self.text_column])
    
    def get_recommendations(self, user_input, top_n=5):
        """
        Generate recommendations based on user input.
        
        Args:
            user_input (str): Text description of user preferences
            top_n (int, optional): Number of top recommendations to return
            
        Returns:
            list: List of dictionaries containing recommended items with their scores
        """
        # Transform user input
        user_vector = self.vectorizer.transform([user_input])
        
        # Compute cosine similarity between user input and all items
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top N similar items
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Create recommendations list
        recommendations = []
        for idx in top_indices:
            item = {
                'title': self.df.iloc[idx][self.title_column],
                'similarity': similarities[idx]
            }
            
            # Add additional columns if needed
            for col in self.df.columns:
                if col != self.title_column and col != self.text_column:
                    item[col] = self.df.iloc[idx][col]
            
            recommendations.append(item)
        
        return recommendations
    
    def print_recommendations(self, recommendations):
        """
        Pretty-print the recommendations.
        
        Args:
            recommendations (list): List of recommendation dictionaries
        """
        print("\nTop Recommendations:")
        print("-" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Similarity Score: {rec['similarity']:.4f}")
            
            # Print additional info if available
            for key, value in rec.items():
                if key not in ['title', 'similarity']:
                    print(f"   {key.capitalize()}: {value}")
            
            print("-" * 50)

def main():
    """Main function to run the recommendation system from command line"""
    # Check if command line arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python recommend.py \"Your query here\"")
        return
    
    # Get query from command line
    query = sys.argv[1]
    
    # Initialize recommender with movie dataset
    # Change the path and column names according to your dataset
    recommender = ContentBasedRecommender(
        dataset_path='movies_data.csv',
        text_column='overview',
        title_column='title'
    )
    
    # Get recommendations
    recommendations = recommender.get_recommendations(query, top_n=5)
    
    # Print recommendations
    recommender.print_recommendations(recommendations)

if __name__ == "__main__":
    main()