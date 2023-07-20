import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import StandardScaler


# Step 1: Load the example dataset
url='https://raw.githubusercontent.com/emreyazicicode/code2023/main/week9/week9_music_train.csv'
df=pd.read_csv(url)
# df=pd.DataFrame(data)



df = df.dropna(how='any')

# Extract features based on specific conditions
df['IsSpecialArtist'] = df['Artist Name'].apply(lambda x: int(x.startswith('&') and len(x.split()) >= 2))
df['NumWords'] = df['Artist Name'].apply(lambda x: len(x.split()))

# Perform one-hot encoding on the "ArtistName" column
# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# artist_encoded = encoder.fit_transform(df[['Artist Name']])
# df = pd.concat([df, pd.DataFrame(artist_encoded, columns=encoder.get_feature_names_out(['Artist Name']))], axis=1)

# Spliting test and train data
X = df.drop(["Class", "Artist Name",'Track Name'],axis=1)
y = df["Class"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Resample the data to address class imbalance
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# Split the resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25)

# Train the random forest model
rf = RandomForestClassifier(n_estimators=200, max_depth=11)
rf.fit(X_train, y_train)

# Evaluate the model performance
y_pred = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)

print('Accuarcy of over sampling:', accuracy)

