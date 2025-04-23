import pandas as pd
import os

class CareerChatbot:
    def __init__(self):
        # Use relative paths, or handle multiple possible locations with try/except
        try:
            # First try with the expected relative path
            self.universities = pd.read_csv('recommender-data/raw/career_university_dataset.csv')
            self.similar_careers = pd.read_csv('recommender-data/raw/similar_careers_dataset.csv')
        except FileNotFoundError:
            # If that fails, try with a path relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)  # Go up one directory
            
            self.universities = pd.read_csv(
                os.path.join(base_dir, 'recommender-data/raw/career_university_dataset.csv')
            )
            self.similar_careers = pd.read_csv(
                os.path.join(base_dir, 'recommender-data/raw/similar_careers_dataset.csv')
            )

    def recommend(self, gpa, predicted_career):
        # Filter matching universities based on GPA and career
        uni_matches = self.universities[
            (self.universities['Career_Field'] == predicted_career) &
            (self.universities['Min_GPA_100'] <= gpa) &
            (self.universities['Max_GPA_100'] >= gpa)
        ][['University_Name', 'Rank_Tier']].to_dict(orient='records')

        # Fetch similar careers
        similar_row = self.similar_careers[self.similar_careers['Career_Field'] == predicted_career]
        similar_list = similar_row['Similar_Careers'].iloc[0].split(", ") if not similar_row.empty else []

        return uni_matches, similar_list
