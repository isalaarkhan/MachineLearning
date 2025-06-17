import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

# Interaction: Class, Age, Fare, Survival
sns.pairplot(titanic[['age', 'fare', 'class', 'survived']], hue='survived')
plt.suptitle('Multivariate Analysis', y=1.02)
plt.show()