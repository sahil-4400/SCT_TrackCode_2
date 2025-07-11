import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset from a CSV file into a pandas DataFrame
df = pd.read_csv("E:\\lovely\\internship\\tasks\\task-2\\Titanic-Dataset.csv")

# Display initial data overview
print("### Initial Data Overview")
print("#### First 5 rows of the dataset:")
print(df.head())  
print("\n#### Dataset Information:")
df.info()  
print("\n#### Missing Values before Cleaning:")
print(df.isnull().sum()) 

# --- Data Cleaning ---

# 1. Handle Missing Values
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# 'Embarked': Fill missing 'Embarked' values with the most frequent port (mode)
most_frequent_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_frequent_embarked, inplace=True)

# 'Cabin': Drop the 'Cabin' column due to excessive missing values (over 77% missing)
df.drop('Cabin', axis=1, inplace=True)

# 2. Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# 3. Convert categorical features to numerical if needed for some analyses

# Display summary after data cleaning
print("\n### Data Cleaning Summary")
print("#### Missing Values after Cleaning:")
print(df.isnull().sum())  
print("\n#### Data types after cleaning:")
df.info()  

# --- Exploratory Data Analysis (EDA) ---

print("\n### Exploratory Data Analysis (EDA)")

# 1. Distribution of Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='viridis')
plt.title('Survival Distribution (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Number of Passengers')
plt.show()

# Calculate and display survival rate
survival_rate = df['Survived'].value_counts(normalize=True) * 100
print(f"\nSurvival Rate:\n{survival_rate}")

# 2. Survival by Gender
plt.figure(figsize=(7, 5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Display survival rate by gender
print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack() * 100)

# 3. Survival by Passenger Class (Pclass)
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='plasma')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Display survival rate by passenger class
print("\nSurvival Rate by Passenger Class:")
print(df.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack() * 100)

# 4. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='purple')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# 5. Survival by Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(x='Age', hue='Survived', data=df, bins=30, kde=True, palette='viridis')
plt.title('Survival Distribution by Age')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# 6. Survival by Embarked Port
plt.figure(figsize=(7, 5))
sns.countplot(x='Embarked', hue='Survived', data=df, palette='rocket')
plt.title('Survival Count by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Display survival rate by embarked port
print("\nSurvival Rate by Embarked Port:")
print(df.groupby('Embarked')['Survived'].value_counts(normalize=True).unstack() * 100)

# 7. Distribution of Fare
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], bins=50, kde=True, color='green')
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()

# 8. Survival by Fare (using a box plot for better comparison)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=df, palette='pastel')
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Fare')
plt.show()

# 9. Relationship between SibSp (siblings/spouses aboard) and Survival
plt.figure(figsize=(8, 5))
sns.countplot(x='SibSp', hue='Survived', data=df, palette='mako')
plt.title('Survival Count by Number of Siblings/Spouses Aboard')
plt.xlabel('Number of Siblings/Spouses (SibSp)')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Display survival rate by SibSp
print("\nSurvival Rate by SibSp:")
print(df.groupby('SibSp')['Survived'].value_counts(normalize=True).unstack() * 100)

# 10. Relationship between Parch (parents/children aboard) and Survival
plt.figure(figsize=(8, 5))
sns.countplot(x='Parch', hue='Survived', data=df, palette='flare')
plt.title('Survival Count by Number of Parents/Children Aboard')
plt.xlabel('Number of Parents/Children (Parch)')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Display survival rate by Parch
print("\nSurvival Rate by Parch:")
print(df.groupby('Parch')['Survived'].value_counts(normalize=True).unstack() * 100)

# 11. Correlation Heatmap 
df_encoded = df.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded = pd.get_dummies(df_encoded, columns=['Embarked'], drop_first=True)

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()
