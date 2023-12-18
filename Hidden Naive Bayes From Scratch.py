import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('heart.csv')
class HiddenNaiveBayesHeartDisease:
    def _init_(self, num_classes, num_hidden_states, num_features):
        self.num_classes = num_classes
        self.num_hidden_states = num_hidden_states
        self.num_features = num_features
        self.class_probs = np.zeros(num_classes)
        self.transition_probs = np.zeros((num_classes, num_hidden_states, num_hidden_states))
        self.observation_probs = {i: {col: 0 for col in df.columns[2:]} for i in range(num_classes)}

    def train(self, data, labels):
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Implement training logic to estimate parameters
        for label in range(self.num_classes):
            # Filter data for the current class
            class_data = data_train[labels_train == label]

            # Estimate class probabilities
            self.class_probs[label] = len(class_data) / len(data_train)

            # Estimate transition probabilities
            for i in range(self.num_hidden_states):
                for j in range(self.num_hidden_states):
                    # Use pandas boolean indexing to filter data
                    count_i_j = len(class_data[(class_data.iloc[:, 0] == i) & (class_data.iloc[:, 1] == j)])
                    total_count_i = len(class_data[class_data.iloc[:, 0] == i])

                    # Avoid division by zero
                    if total_count_i == 0:
                        self.transition_probs[label, i, j] = 0
                    else:
                        self.transition_probs[label, i, j] = count_i_j / total_count_i

            # Estimate observation probabilities
            for i in range(self.num_hidden_states):
                for col_name in data.columns[2:]:
                    self.observation_probs[label][col_name] = np.mean(class_data[class_data.iloc[:, 0] == i][col_name])

    def predict(self, features):
      
        # Implement inference logic to predict class labels
        predictions = []
        for instance_features in features:
            instance_probs = []
            for label in range(self.num_classes):
                class_prob = np.log(self.class_probs[label])
                transition_prob = 0
                observation_prob = 0
                for i in range(self.num_hidden_states):
                    # Handle the case where the feature value is a string (e.g., 'A')
                    try:
                        feature_value = instance_features[i]
                        # Assuming you have the test data as a list of lists
                        transition_prob += np.log(self.transition_probs[label, int(feature_value), i])
                        for col_name in data.columns[2:]:
                            observation_prob += np.log(np.random.normal(self.observation_probs[label][col_name], 1e-3))
                    except (ValueError, IndexError):
                        # Handle the conversion to int or index error (if the feature value is not an integer)
                        transition_prob += 0

                instance_probs.append(class_prob + transition_prob + observation_prob)
            predictions.append(np.argmax(instance_probs))
        return np.array(predictions)

# Example usage:
# Assuming you have heart disease data (heart_data) and corresponding labels (heart_labels)
num_classes = 2
num_hidden_states = 3
num_features = 20  # Adjust this based on your heart disease dataset features

heart_nb_model = HiddenNaiveBayesHeartDisease(num_classes, num_hidden_states, num_features)
target='Heart Attack Risk'
heart_data=df.drop(target,axis=1)
heart_labels=df[targe…