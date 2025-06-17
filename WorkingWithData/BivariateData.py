import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

# Age vs. Fare colored by survival
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic)
plt.title('Bivariate: Age vs Fare')
plt.show()

# Survival rate by class
sns.barplot(x='class', y='survived', data=titanic)
plt.title('Bivariate: Class vs Survival')
plt.show()