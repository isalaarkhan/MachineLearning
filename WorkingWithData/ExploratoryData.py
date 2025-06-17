import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

# Age distribution
sns.histplot(titanic['age'], kde=True)
plt.title('Univariate: Age Distribution')
plt.show()

# Survival count
sns.countplot(x='survived', data=titanic)
plt.title('Univariate: Survival Count')
plt.show()