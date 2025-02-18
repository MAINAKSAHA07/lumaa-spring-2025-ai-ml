from content_recommender import ContentBasedRecommender

# Sample queries to test
sample_queries = [
    "I love thrilling action movies set in space, with a comedic twist.",
    "I prefer heartwarming dramas with strong female leads.",
    "I want to see documentaries about nature and wildlife."
]

# Initialize recommender
recommender = ContentBasedRecommender(
    dataset_path='movies_data.csv',
    text_column='overview',
    title_column='title'
)

# Test each query
for query in sample_queries:
    print(f"\nQuery: {query}")
    recommendations = recommender.get_recommendations(query, top_n=3)
    recommender.print_recommendations(recommendations)
    print("\n" + "="*60 + "\n")