import pandas as pd

class CareerChatbot:
    def __init__(self):
        self.universities = pd.read_csv('recommender-data/raw/career_university_dataset.csv')
        self.similar_careers = pd.read_csv('recommender-data/raw/similar_careers_dataset.csv')

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
