import scipy.stats as stats
import pandas as pd

df1 = pd.read_csv('PaLMF.csv');
df2 = pd.read_csv('PaLMF+.csv');
df3 = pd.read_csv('PaLMF++.csv')

statistic, p_value = stats.mannwhitneyu(df1, df2, alternative='two-sided')
print(p_value)
statistic, p_value = stats.mannwhitneyu(df1, df3, alternative='two-sided')
print(p_value)
statistic, p_value = stats.mannwhitneyu(df2, df3, alternative='two-sided')
print(p_value)
