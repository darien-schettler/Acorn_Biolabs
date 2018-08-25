import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')
# --------------------------------------------------------------------------

x = np.linspace(-np.pi, np.pi, 10)
y = np.sin(x)
plt.plot(x, y, color='r', marker='x', linestyle=':', label='Sin Wave')
plt.show()

input("\n\nPress Enter to continue...")

print("\n\nPlot Attributes\n\n"
      "• color\n"
      "• marker\n"
      "• linestyle\n"
      "• label\n"
      "• title\n"
      "• grid\n"
      "• legend\n")

df = pd.read_csv("mtcars.csv")
print(df.columns)
plt.scatter(df.mpg, df.wt)
plt.show()

input("\n\nPress Enter to continue...")

plt.plot(df[['mpg', 'cyl', 'wt']])
plt.show()

input("\n\nPress Enter to continue...")

df['mpg'].plot(kind='bar')
plt.show()

input("\n\nPress Enter to continue...")

x = [1, 2, 3, 4, 0.5]
plt.pie(x)
plt.show()

input("\n\nPress Enter to continue...")

plt.hist(df['hp'])
plt.show()

input("\n\nPress Enter to continue...")

df.plot(kind='scatter', x='hp', y='mpg', c=['darkgray'], s=150)
sb.regplot(x='hp', y='mpg', data=df, scatter=True)
plt.show()

input("\n\nPress Enter to continue...")

sb.pairplot(df[['mpg', 'hp', 'wt']])
plt.show()

input("\n\nPress Enter to continue...")

df.boxplot(column='hp', by='am')
df.boxplot(column='mpg', by='am')
plt.savefig('test.png')
plt.show()

input("\n\nPress Enter to continue...")

x = range(1, 10)
z = range(-9, 10, 2)
y = [1, 2, 3, 4, 0, 4, 3, 2, 1]
fig = plt.figure()

# [x_pos, y_pos, x_stretch, y_stretch]
ax = fig.add_axes([0.1, 0.1, 0.83, 0.83])
ax.plot(x, y, z)
plt.show()

input("\n\nPress Enter to continue...")


# --------------------------------------------------------------------------
