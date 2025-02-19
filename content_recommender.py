import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class MovieRecommender:
    def __init__(self, dataset_path):
        """
        Initialize the movie recommender system.
        
        Args:
            dataset_path (str): Path to the movies CSV file
        """
        # Load and preprocess the dataset
        self.df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(self.df)} movies")
        
        # Extract movie descriptions from crew data
        self.df['description'] = self.df.apply(self._create_movie_description, axis=1)
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['description'])
        print("Vectorization complete")

    def _create_movie_description(self, row):
        """
        Create a text description for a movie using available information.
        
        Args:
            row: DataFrame row containing movie information
        
        Returns:
            str: Combined movie description
        """
        # Extract cast information
        try:
            cast_data = json.loads(row['cast'])
            cast_names = ' '.join([person['name'] for person in cast_data[:5] 
                                 if 'name' in person])
        except:
            cast_names = ''
        
        # Extract crew information
        try:
            crew_data = json.loads(row['crew'])
            crew_info = ' '.join([f"{person.get('job', '')} {person.get('name', '')}" 
                                for person in crew_data 
                                if person.get('job') in ['Director', 'Writer', 'Producer']])
        except:
            crew_info = ''
        
        # Combine all information
        description = f"{row['title']} {cast_names} {crew_info}"
        return description.lower()

    def get_recommendations(self, query, n=5):
        """
        Get movie recommendations based on the input query.
        
        Args:
            query (str): User's movie preference description
            n (int): Number of recommendations to return
        
        Returns:
            list: Top N movie recommendations
        """
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top N movie indices
        top_indices = similarity_scores.argsort()[-n:][::-1]
        
        # Create recommendations list
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'title': self.df.iloc[idx]['title'],
                'similarity': similarity_scores[idx],
                'cast': self._get_cast_names(self.df.iloc[idx]['cast'])
            })
        
        return recommendations

    def _get_cast_names(self, cast_json):
        """Extract cast names from JSON string"""
        try:
            cast_data = json.loads(cast_json)
            return ', '.join([person['name'] for person in cast_data[:3] 
                            if 'name' in person])
        except:
            return ''

def main():
    # Initialize recommender
    recommender = MovieRecommender('movies_data.csv')
    
    # Get user input
    query = input("Enter your movie preferences: ")
    
    # Get recommendations
    recommendations = recommender.get_recommendations(query)
    
    # Print recommendations
    print("\nRecommended Movies:")
    print("-" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Similarity Score: {rec['similarity']:.4f}")
        print(f"   Cast: {rec['cast']}")
        print("-" * 50)

if __name__ == "__main__":
    main()