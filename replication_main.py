import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS,IVLIML,IVGMM
from numpy.linalg import inv,pinv
import statsmodels.formula.api as smf

# Importing Data Files
df_county=pd.read_stata('miansufieconometrica_countylevel.dta')
df_industry_county=pd.read_stata('miansufieconometrica_countyindustrylevel.dta')


# Figures

# Figure 1 - Housing Net Worth Shock and Non-tradable Employment
# Load data
data = df_county

# Filter the data
filtered_data_1 = data[(data['total'] > 50_000) & (data['CCemp2_0709_2'] < 0.2) & (data['netwp_h'] >= -0.3)]
filtered_data_2 = data[(data['total'] > 50_000) & (data['CH2emp2_0709_1'].abs() < 0.2) & (data['netwp_h'] >= -0.3)]

# Function to plot scatter, linear fit, and LOWESS
def plot_graph(x, y, title, ylabel, filename):
    plt.figure(figsize=(6, 4))
    # Scatter plot
    sns.scatterplot(x=x, y=y, data=filtered_data_1, s=10, color='blue', label="Data")
    # Linear fit
    X = sm.add_constant(filtered_data_1[x])
    model = sm.OLS(filtered_data_1[y], X).fit()
    #plt.plot(filtered_data_1[x], model.predict(X), color='black', linewidth=2, label="Linear Fit")
    plt.plot(filtered_data_1[x], model.predict(X), color='red', linewidth=2, label="Linear Fit")
    # LOWESS smoothing
    lowess = sm.nonparametric.lowess(filtered_data_1[y], filtered_data_1[x], frac=0.3)
    plt.plot(lowess[:, 0], lowess[:, 1], linestyle='solid', color='black', label="LOWESS")
    plt.xlabel("Change in housing net worth, 2006-2009")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    #plt.show()

# Create the graphs
plot_graph("netwp_h", "CCemp2_0709_2", "Non-Tradable Employment Growth", "Non-Tradable Employment Growth 07Q1-09Q1\n(restaurants & retail)", "Figure1a.png")
plot_graph("netwp_h", "CH2emp2_0709_1", "Non-Tradable Sector Employment Growth", "Non-Tradable Sector Employment Growth 07Q1-09Q1\n(based on low geographical concentration)", "Figure1b.png")

# Combine the two plots
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
img1 = plt.imread("Figure1a.png")
img2 = plt.imread("Figure1b.png")
axes[0].imshow(img1)
axes[1].imshow(img2)
for ax in axes:
    ax.axis("off")
plt.savefig("Figure1.png")





# Tables

####################################################################################
# Table 1 - Industry Categorization
####################################################################################
# Load the data
data = df_industry_county
data['naics']=data['naics'].apply(lambda x: x[:4])
# Keep only one observation per 'naics'
data = data.drop_duplicates(subset=['naics'])
# Keep observations where Iemp2_2007 is not missing
data = data[data['Iemp2_2007'].notna()]
# Compute total employment
data['totalemp'] = data['Iemp2_2007'].sum()
# Compute employment share
data['Iempshare07'] = (data['Iemp2_2007'] / data['totalemp']) * 100
# Select relevant columns
data = data[['indcat', 'nontradable_strict', 'naics', 'industry', 'Iempshare07', 'Ihcat2']]
# Sort data
data = data.sort_values(by=['indcat', 'nontradable_strict', 'Iempshare07'], ascending=[True, True, False])
# Filter data by each goods category and breakdown employment share by NAICS
writer = pd.ExcelWriter('Table1.xlsx')
for cat in data.indcat.unique():
  cat_slice = data[data.indcat==cat]
  cat_slice.index.name=f'{cat}'
  cat_slice_2 = cat_slice[['naics','industry','Iempshare07']]
  cat_slice_2=cat_slice_2.copy()
  cat_slice_2.rename(columns={'Iempshare07': '%','industry':'Industry name'}, inplace=True)
  cat_slice_2['%']=cat_slice_2['%'].round(2)
  cat_slice_2.loc[-1]={'naics':None,'Industry name':'Total','%':round(cat_slice_2['%'].sum(),1)}
  # save output as part of an .xlsx workbook
  cat_slice_2.to_excel(writer, sheet_name=f'{cat}')
writer.close()
####################################################################################
# Table 2 - Industry Categorization Based On Geographical Concentration
####################################################################################
writer = pd.ExcelWriter('Table2.xlsx')

# Load the dataset
data = df_industry_county
# Tag each unique NAICS code – keep one row per NAICS
data = data.drop_duplicates(subset="naics").copy()
# Create `tradable` and `nontradable` indicators
data["tradable"] = (data["indcat"] == 1).astype(int)
data["nontradable"] = (data["indcat"] == 2).astype(int)

# Sort descending by Iherf
data_top_30 = data.sort_values(by="Iherf", ascending=False)
# Keep relevant columns
data_top_30 = data_top_30[["tradable", "naics", "industry"]].head(30)
data_top_30.index.name='Top 30'
data_top_30.to_excel(writer, sheet_name='Top 30')
# Sort ascending by Iherf
data_bottom_30 = data.sort_values(by="Iherf", ascending=True)
# Keep relevant columns
data_bottom_30 = data_bottom_30[[ "tradable", "naics", "industry"]].head(30)
data_bottom_30.index.name='Bottom 30'
data_bottom_30.to_excel(writer, sheet_name='Bottom 30')

writer.close()
####################################################################################
# Table 3 - Summary Statistics
####################################################################################
# Load the county-level data
data = df_county
# Keep only non-missing values for netwp_h
data = data[data['netwp_h'].notna()]

# Unweighted summary statistics
columns_unweighted = [
    "netwp_h", "total", "Clf_0709", "Cemp2_2007", "Cemp2_0709", "Cwage_2007", "Cwage_0709", "elasticity",
    "CCemp2_0709_2", "CFemp2_0709", "CCemp2_0709_1", "CCemp2_0709_3", "CCemp2_0709_0",
    "Cwagehr_Wmean2007", "Cwagehr_p102007", "Cwagehr_p252007", "Cwagehr_p752007", "Cwagehr_p752007", "Cwagehr_p902007","Cwagehr_median2007",
    "Cwagehr_p100709", "Cwagehr_p250709", "Cwagehr_p750709", "Cwagehr_p900709","Cwagehr_median0709", "Cwagehr_Wmean0709"
]
unweighted_stats = data[columns_unweighted].describe(percentiles=[0.1, 0.9])

# Additional tabstat computations for specific variables
extra_columns = ["CCemp2_0709_2", "CFemp2_0709", "CCemp2_0709_1", "CCemp2_0709_3", "CCemp2_0709_0"]
extra_stats = data[extra_columns].describe(percentiles=[0.1, 0.9])

# Weighted means and standard deviations
weighted_columns = columns_unweighted
weighted_stats = data[weighted_columns].apply(lambda x: pd.Series({
    "mean": (x * data["total"]).sum() / data["total"].sum(),
    "std": ((x - (x * data["total"]).sum() / data["total"].sum())**2 * data["total"]).sum() / data["total"].sum()
}))

# Industry-level data
industry_data = df_industry_county
industry_data = industry_data.drop_duplicates(subset=['naics'])

# Industry-level summary statistics
industry_stats = industry_data['Iherf'].describe(percentiles=[0.1, 0.9])
industry_weighted_stats = pd.Series({
    "mean": (industry_data["Iherf"] * industry_data["Iemp2_2007"]).sum() / industry_data["Iemp2_2007"].sum(),
    "std": ((industry_data["Iherf"] - (industry_data["Iherf"] * industry_data["Iemp2_2007"]).sum() / industry_data["Iemp2_2007"].sum())**2 * industry_data["Iemp2_2007"]).sum() / industry_data["Iemp2_2007"].sum()
})

unweighted_stats.rename(columns={'netwp_h':'Housing net worth shock, 2006-2009',
 'total':'Number of households, 2000',
 'Clf_0709':'Labor force growth, 2007 to 2009',
 'Cemp2_2007':'Total employment, 2007 ',
 'Cemp2_0709':'Employment growth, 2007 to 2009',
 'Cwage_2007':'Average wage, 2007 ',
 'Cwage_0709':'Average wage growth, 2007 to 2009',
 'elasticity':'Housing supply elasticity (Saiz)',
 'CCemp2_0709_2':'Non-tradable employment growth, 2007 to 2009',
 'CFemp2_0709':'Food industry employment growth, 2007 to 2009',
 'CCemp2_0709_1':'Tradable employment growth, 2007 to 2009',
 'CCemp2_0709_3':'Construction employment growth, 2007 to 2009',
 'CCemp2_0709_0':'Other employment growth, 2007 to 2009',
 'Cwagehr_Wmean2007':'Industry geographical herfindahl, 2007',
 'Cwagehr_p102007':'Hourly wage, 2007 ',
 'Cwagehr_p252007':'Hourly wage, 10th percentile, 2007',
 'Cwagehr_p752007':'Hourly wage, 25th percentile, 2007',
 'Cwagehr_p752007':'Hourly wage, median, 2007',
 'Cwagehr_p902007':'Hourly wage, 75th percentile, 2007',
 'Cwagehr_median2007':'Hourly wage, 90th percentile, 2007',
 'Cwagehr_p100709':'Wage growth, 2007 to 2009',
 'Cwagehr_p250709':'Wage growth, 10th percentile, 2007-09',
 'Cwagehr_p750709':'Wage growth, 25th percentile, 2007 to 2009',
 'Cwagehr_p900709':'Wage growth, median, 2007 to 2009',
 'Cwagehr_median0709':'Wage growth, 75th percentile, 2007 to 2009',
 'Cwagehr_Wmean0709':'Wage growth, 90th percentile, 2007 to 2009'},inplace=True)


weighted_stats.rename(columns={'netwp_h':'Housing net worth shock, 2006-2009',
 'total':'Number of households, 2000',
 'Clf_0709':'Labor force growth, 2007 to 2009',
 'Cemp2_2007':'Total employment, 2007 ',
 'Cemp2_0709':'Employment growth, 2007 to 2009',
 'Cwage_2007':'Average wage, 2007 ',
 'Cwage_0709':'Average wage growth, 2007 to 2009',
 'elasticity':'Housing supply elasticity (Saiz)',
 'CCemp2_0709_2':'Non-tradable employment growth, 2007 to 2009',
 'CFemp2_0709':'Food industry employment growth, 2007 to 2009',
 'CCemp2_0709_1':'Tradable employment growth, 2007 to 2009',
 'CCemp2_0709_3':'Construction employment growth, 2007 to 2009',
 'CCemp2_0709_0':'Other employment growth, 2007 to 2009',
 'Cwagehr_Wmean2007':'Industry geographical herfindahl, 2007',
 'Cwagehr_p102007':'Hourly wage, 2007 ',
 'Cwagehr_p252007':'Hourly wage, 10th percentile, 2007',
 'Cwagehr_p752007':'Hourly wage, 25th percentile, 2007',
 'Cwagehr_p752007':'Hourly wage, median, 2007',
 'Cwagehr_p902007':'Hourly wage, 75th percentile, 2007',
 'Cwagehr_median2007':'Hourly wage, 90th percentile, 2007',
 'Cwagehr_p100709':'Wage growth, 2007 to 2009',
 'Cwagehr_p250709':'Wage growth, 10th percentile, 2007-09',
 'Cwagehr_p750709':'Wage growth, 25th percentile, 2007 to 2009',
 'Cwagehr_p900709':'Wage growth, median, 2007 to 2009',
 'Cwagehr_median0709':'Wage growth, 75th percentile, 2007 to 2009',
 'Cwagehr_Wmean0709':'Wage growth, 90th percentile, 2007 to 2009'},inplace=True)

unweighted_stats=unweighted_stats.T
weighted_stats=weighted_stats.T
unweighted_stats.rename(columns={'count':'N',	'mean':'Mean',	'std':'SD','10%':'10th','50%':'Median','90%':'90th'},inplace=True)
weighted_stats.rename(columns={'mean':'Weighted mean','std':'Weighted SD'},inplace=True)

table_3 = pd.concat([unweighted_stats,weighted_stats],axis=1)
table_3=table_3.round(3)
table_3.to_excel('Table3.xlsx')

####################################################################################
#Table 4 - Non-Tradable Employment Growth And The Housing Net Worth Shock
####################################################################################
data = df_county

# List of shock variable columns
shock_vars = [f"C2D06share{i}" for i in range(1, 24)]
# Drop missing values for needed variables
data.dropna(subset=["CCemp2_0709_2", "CH2emp2_0709_1", "netwp_h", "total", "statename"],inplace=True)

# Col 1: Non-tradable definition used: Restaurant & Retail, Specification: OLS, 2-digit 2006 employment share controls included?: N
y1 = data["CCemp2_0709_2"]
X1 = sm.add_constant(data["netwp_h"])
wls1 = sm.WLS(y1, X1, weights=data["total"])
res1 = wls1.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
#print(res1.summary())
#print('-----------------------------')
# Col 2: Non-tradable definition used: Geographical Concentration, Specification: OLS, 2-digit 2006 employment share controls included?: N
y2 = data["CH2emp2_0709_1"]
X2 = sm.add_constant(data["netwp_h"])
wls2 = sm.WLS(y2, X2, weights=data["total"])
res2 = wls2.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
#print('-----------------------------')
# Col 3: Non-tradable definition used: Employment Growth Restaurant & Retail, Specification: IV, 2-digit 2006 employment share controls included?: N
data_iv = data.dropna(subset=["elasticity"])
exog = sm.add_constant(data_iv[shock_vars[:-1] ])
exog = exog.iloc[:,0]
endog = data_iv["netwp_h"]
instr = data_iv["elasticity"]
y_iv = data_iv["CCemp2_0709_2"]
model_iv_3 = IV2SLS(dependent=y_iv, exog=exog, endog=endog, instruments=instr, weights=data_iv['total'])
results_iv_3 = model_iv_3.fit(cov_type='clustered', clusters=data_iv["statename"])
#print(results_iv_3.summary)
#print('-----------------------------')
# Col 4: Non-tradable definition used: Employment Growth Geographical Concentration, Specification: IV, 2-digit 2006 employment share controls included?: N
data_iv = data.dropna(subset=["elasticity"])
exog = sm.add_constant(data_iv[shock_vars[:-1] ])
exog = exog.iloc[:,0]
endog = data_iv["netwp_h"]
instr = data_iv["elasticity"]
y_iv = data_iv["CH2emp2_0709_1"]
model_iv_4 = IV2SLS(dependent=y_iv, exog=exog, endog=endog, instruments=instr, weights=data_iv['total'])
results_iv_4 = model_iv_4.fit(cov_type='clustered', clusters=data_iv["statename"])
#print(results_iv_4.summary)
#print('-----------------------------')
# Col 5: Non-tradable definition used: Employment Growth Restaurant & Retail, Specification: OLS, 2-digit 2006 employment share controls included?: Y
y5 = data["CCemp2_0709_2"]
X5 = sm.add_constant(data[["netwp_h"] + shock_vars[:-1]])
wls5 = sm.WLS(y5, X5, weights=data["total"])
res5 = wls5.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
#print(res5.summary())
#print('-----------------------------')
# Col 6: Non-tradable definition used: Employment Growth Geographical Concentration, Specification: OLS, 2-digit 2006 employment share controls included?: Y
y6 = data["CH2emp2_0709_1"]
X6 = sm.add_constant(data[["netwp_h"] + shock_vars[:-1]])
wls6 = sm.WLS(y6, X6, weights=data["total"])
res6 = wls6.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
#print(res6.summary())
#print('-----------------------------')
# Col 7: Non-tradable definition used: Restaurant & Retail, Specification: IV, 2-digit 2006 employment share controls included?: Y
data_iv = data.dropna(subset=["elasticity"])
exog = sm.add_constant(data_iv[shock_vars[:-1] ])
endog = data_iv["netwp_h"]
instr = data_iv["elasticity"]
y_iv = data_iv["CCemp2_0709_2"]
model_iv_7 = IV2SLS(dependent=y_iv, exog=exog, endog=endog, instruments=instr, weights=data_iv['total'])
results_iv_7 = model_iv_7.fit(cov_type='clustered', clusters=data_iv["statename"])
#print(results_iv_7.summary)
#print('-----------------------------')
# Col 8: Non-tradable definition used: Geographical Concentration, Specification: IV, 2-digit 2006 employment share controls included?: Y
data_iv = data.dropna(subset=["elasticity"])
exog = sm.add_constant(data_iv[shock_vars[:-1] ])
endog = data_iv["netwp_h"]
instr = data_iv["elasticity"]
y_iv = data_iv["CH2emp2_0709_1"]
model_iv_8 = IV2SLS(dependent=y_iv, exog=exog, endog=endog, instruments=instr, weights=data_iv['total'])
results_iv_8 = model_iv_8.fit(cov_type='clustered', clusters=data_iv["statename"])
#print(results_iv_8.summary)
#print('-----------------------------')

# Create MultiIndex for rows
index = pd.MultiIndex.from_tuples([('Change in Housing Net Worth,2006-2009', 'coef.'), ('Change in Housing Net Worth,2006-2009', 'std. err'), ('Change in Housing Net Worth,2006-2009', 'p value'),
                                    ('Constant', 'coef.'), ('Constant', 'std. err'),('Constant','p value'),
                                     ('N',''),
                                   ('R-squared','')] ) #names=['Group', 'Number']
# Create multilevel columns
columns = pd.MultiIndex.from_tuples([('Restaurant & Retail', 'OLS','N'),
                                    ('Geographical Concentration', 'OLS','N'),
                                    ('Restaurant & Retail', 'IV','N'),
                                    ('Geographical Concentration', 'IV','N'),
                                     ('Restaurant & Retail', 'OLS','Y'),
                                     ('Geographical Concentration', 'OLS','Y'),
                                     ('Restaurant & Retail', 'IV','Y'),
                                     ('Geographical Concentration', 'IV','Y')],
                                    names=['Non-tradable definition used','Specification','Controls?']) #names=['Metric', 'Sub']
# Create the DataFrame
data = [[round(res1.params['netwp_h'],3), round(res2.params['netwp_h'],3), round(results_iv_3.params['netwp_h'],3), round(results_iv_4.params['netwp_h'],3), round(res5.params['netwp_h'],3), round(res6.params['netwp_h'],3), round(results_iv_7.params['netwp_h'],3), round(results_iv_8.params['netwp_h'],3)],
        [round(res1.bse['netwp_h'],3), round(res2.bse['netwp_h'],3), round(results_iv_3.std_errors['netwp_h'],3), round(results_iv_4.std_errors['netwp_h'],3),round(res5.bse['netwp_h'],3), round(res6.bse['netwp_h'],3), round(results_iv_7.std_errors['netwp_h'],3), round(results_iv_8.std_errors['netwp_h'],3)],
        [round(res1.pvalues['netwp_h'],3), round(res2.pvalues['netwp_h'],3), round(results_iv_3.pvalues['netwp_h'],3), round(results_iv_4.pvalues['netwp_h'],3),round(res5.pvalues['netwp_h'],3), round(res6.pvalues['netwp_h'],3), round(results_iv_7.pvalues['netwp_h'],3), round(results_iv_8.pvalues['netwp_h'],3)],
        [round(res1.params['const'],3), round(res2.params['const'],3), round(results_iv_3.params['const'],3), round(results_iv_4.params['const'],3),round(res5.params['const'],3), round(res6.params['const'],3), round(results_iv_7.params['const'],3), round(results_iv_8.params['const'],3)],
        [round(res1.bse['const'],3), round(res2.bse['const'],3), 19, 20,round(res5.bse['const'],3), round(res6.bse['const'],3), round(results_iv_7.std_errors['const'],3), round(results_iv_8.std_errors['const'],3)],
        [round(res1.pvalues['const'],3), round(res2.pvalues['const'],3), round(results_iv_3.std_errors['const'],3), round(results_iv_4.std_errors['const'],3),round(res5.pvalues['const'],3), round(res6.pvalues['const'],3), round(results_iv_7.pvalues['const'],3), round(results_iv_8.pvalues['const'],3)],
        [int(res1.nobs), int(res2.nobs), int(results_iv_3.nobs), int(results_iv_4.nobs),int(res5.nobs), int(res6.nobs), int(results_iv_7.nobs), int(results_iv_8.nobs)],
        [round(res1.rsquared,3), round(res2.rsquared,3), round(results_iv_3.rsquared,3), round(results_iv_4.rsquared,3),round(res5.rsquared,3), round(res6.rsquared,3), round(results_iv_7.rsquared,3), round(results_iv_8.rsquared,3)]]
table_4 = pd.DataFrame(data, index=index, columns=columns)
table_4.to_excel('Table4.xlsx')

####################################################################################
#Table 5 - Is Non-Tradable Employment Growth Driven By Construction Sector Shock?
####################################################################################



####################################################################################
#Table 6 - Is Non-Tradable Employment Growth Driven By Credit Supply Tightening?
####################################################################################
writer = pd.ExcelWriter('Table6.xlsx')

data = df_county

dep_vars = [
    "CCemp2_1_4_0709_2",
    "CCemp2_5_9_0709_2",
    "CCemp2_10_19_0709_2",
    "CCemp2_20_49_0709_2",
    "CCemp2_50_99_0709_2",
    "CCemp2_100plus_0709_2"
]

# Dictionary to store results
results = {}
ols_results_df = pd.DataFrame()

iv_results = {}
iv_results_df = pd.DataFrame()

sizes = ['1 to 4','5 to 9','10 to 19','20 to 49','50 to 99','100+']

# Loop through each dependent variable
for var in dep_vars:
    # OLS Regression
    df = data[[var, "netwp_h", "total", "statename"]].dropna()
    y = df[var]
    X = sm.add_constant(df["netwp_h"])
    model = sm.WLS(y, X, weights=df["total"])
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["statename"]})
    ols_results_df[var] = [round(result.params['netwp_h'],3), round(result.bse['netwp_h'],3),round(result.pvalues['netwp_h'],3)]

    # IV Regression
    df = data[[var, "netwp_h", "elasticity", "total", "statename"]].dropna()
    formula = f"{var} ~ 1 + [netwp_h ~ elasticity]"
    model = IV2SLS.from_formula(
        formula,
        data=df,
        weights=df["total"]
    ).fit(cov_type="clustered", clusters=df["statename"])
    iv_results_df[var] = [round(model.params['netwp_h'],3), round(model.std_errors['netwp_h'],3), round(model.pvalues['netwp_h'],3)]

#results
ols_results_df.rename(index={0: 'Coefficient', 1: 'Std. Error', 2: 'P-value'}, inplace=True)
ols_results_df.columns = sizes
ols_results_df=ols_results_df.reset_index()
ols_results_df['specification']='OLS'
ols_results_df.set_index(['specification','index'],inplace=True)
ols_results_df.columns.names = ['Establishment Size In Terms Of Number of Employees']
ols_results_df.index.names = ['specification','Change in Housing Net Worth, 2006-2009']

iv_results_df.rename(index={0: 'Coefficient', 1: 'Std. Error', 2: 'P-value'}, inplace=True)
iv_results_df.columns = sizes
iv_results_df=iv_results_df.reset_index()
iv_results_df['specification']='IV'
iv_results_df.set_index(['specification','index'],inplace=True)
iv_results_df.columns.names = ['Establishment Size In Terms Of Number of Employees']
iv_results_df.index.names = ['specification','Change in Housing Net Worth, 2006-2009']

table_6_AB = pd.concat([ols_results_df,iv_results_df])
table_6_AB.to_excel(writer, sheet_name='PanelAB')

df_panelc = data.copy()
df_panelc = df_panelc.replace([np.inf, -np.inf], np.nan)
#df_panelc = df_panelc[df_panelc["netwp_h"].notna()]
df_panelc = df_panelc.dropna(subset=["elasticity", "netwp_h", "total", "statename"])

# Compute median of Clocalshare
median_localshare = df_panelc["Clocalshare"].median()
# Create 'national' and 'local' dummies
df_panelc["national"] = (df_panelc["Clocalshare"] < median_localshare).astype(int)
df_panelc["local"] = (df_panelc["Clocalshare"] >= median_localshare).astype(int)
# First stage: netwp_h ~ elasticity
X_fs = sm.add_constant(df_panelc["elasticity"])
y_fs = df_panelc["netwp_h"]
model_fs = sm.WLS(y_fs, X_fs, weights=df_panelc["total"]).fit(cov_type="cluster", cov_kwds={"groups": df_panelc["statename"]})
df_panelc["netwpp"] = model_fs.fittedvalues
# Regressions by group
results = {}
market_results_df = pd.DataFrame()

for group, label in zip(["national", "local"], ["National", "Local"]):
    subset = df_panelc[df_panelc[group] == 1]

    for var in ["netwp_h", "netwpp"]:
        X = sm.add_constant(subset[var])
        y = subset["CCemp2_0709_2"]
        model = sm.WLS(y, X, weights=subset["total"]).fit(cov_type="cluster", cov_kwds={"groups": subset["statename"]})

        key = f"{label} - {var}"
        results[key] = model
        #print(f"\nResults for {key}")
        #print(model.summary())
        market_results_df[key] = [round(model.params[var],3), round(model.bse[var],3),round(model.pvalues[var],3)]

market_results_df.rename(index={0: 'Coefficient', 1: 'Std. Error', 2: 'P-value'}, inplace=True)
market_results_df.index.names = ['Change in Housing Net Worth, 2006-2009']

columns = pd.MultiIndex.from_tuples([('National', 'OLS'),
                                    ('Local', 'IV'),
                                     ('National', 'OLS'),
                                     ('Local', 'IV')],
                                    names=['Banking Type','Specification']) #names=['Metric', 'Sub']
market_results_df.columns=columns
table_6_C = market_results_df.copy()
table_6_C.to_excel(writer, sheet_name='PanelC')

writer.close()

####################################################################################
#Table 7 - Tradable Employment Growth And The Housing Net Worth Shock
####################################################################################
data = df_county

shock_vars = [f"C2D06share{i}" for i in range(1, 24)] # List of shock variable columns
data["sqrt_total"] = np.sqrt(data["total"])
data = data.dropna(subset=["CCemp2_0709_2", "CH2emp2_0709_1", "netwp_h", "total", "statename"]) # Drop missing values for needed variables
# (1)
y1 = data["CCemp2_0709_1"]
X1 = sm.add_constant(data["netwp_h"])
wls1 = sm.WLS(y1, X1, weights=data["total"])
res1 = wls1.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
# (2)
y2 = data["CH2emp2_0709_4"]
X2 = sm.add_constant(data["netwp_h"])
wls2 = sm.WLS(y2, X2, weights=data["total"])
res2 = wls2.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
# (3)
y3 = data["CCemp2_0709_1"]
X3 = sm.add_constant(data[["netwp_h"]+shock_vars[:-1] ])
wls3 = sm.WLS(y3, X3, weights=data["total"])
res3 = wls3.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
# (4)
y4 = data["CH2emp2_0709_4"]
X4 = sm.add_constant(data[["netwp_h"]+shock_vars[:-1] ])
wls4 = sm.WLS(y4, X4, weights=data["total"])
res4 = wls4.fit(cov_type='cluster', cov_kwds={'groups': data['statename']})
# (5)
data = df_industry_county
    # Interaction term
data["netwpherf_b"] = data["Iherf"] * data["netwp_h"]
    # Drop missing outcome or regressor
data = data.dropna(subset=["netwp_h", "CIemp2_0709"])
    # Identify unique industries and assign a tag
data["indtag"] = data.duplicated(subset=["industry"], keep="first").apply(lambda x: 0 if x else 1)
    # Create weighted percentiles based on Iherf
        # Keep temp only where indtag==1 and Iemp2_2007 is not missing
data["temp"] = np.where((data["indtag"] == 1) & (data["Iemp2_2007"].notna()), 1, 0)
    # Sort by Iherf
data = data.sort_values(by="Iherf").copy()
    # Weighted cumulative sum of Iemp2_2007 by temp
data["runsumemp"] = (data["Iemp2_2007"] * data["temp"]).cumsum()
totsumemp = (data["Iemp2_2007"] * data["temp"]).sum()
data["Ipercentile"] = data["runsumemp"] / totsumemp
    # Create decile groupings (1 to 10)
#data["Ipercentile_10"] = (data["Ipercentile"] * 10).astype(int) + 1
#data["Ipercentile_10"] = data["Ipercentile_10"].clip(upper=10)
    # Run the regression: Table 5, Col 5
reg_data = data.dropna(subset=["CIemp2_2007", "statename"])
res5 = smf.wls(
    formula="CIemp2_0709 ~ netwp_h + Iherf + netwpherf_b",
    data=reg_data,
    weights=reg_data["CIemp2_2007"]
).fit(cov_type="cluster", cov_kwds={"groups": reg_data["statename"]})
# (6)
data = df_industry_county
# Create interaction term: netwpherf = Iherf * netwp_h
data["netwpherf_b"] = data["Iherf"] * data["netwp_h"]
# Drop missing values for regression
data.dropna(subset=["CIemp2_0709", "netwp_h", "Iherf", "netwpherf_b", "CIemp2_2007", "statename","naics"],inplace=True)
res6 = smf.wls(
    #formula="CIemp2_0709 ~ netwp_h + Iherf + netwpherf",
    formula="CIemp2_0709 ~ netwp_h + netwpherf_b -1 + C(naics)",
    data=data,
    weights=data["CIemp2_2007"]
).fit(cov_type="cluster", cov_kwds={"groups": data["statename"]})
# (7)
res7 = smf.wls(
    formula="CIemp2_0709 ~ netwpherf_b -1 + C(naics)+ C(countyname)",
    data=data,
    weights=data["CIemp2_2007"]
).fit(cov_type="cluster", cov_kwds={"groups": data["statename"]})

# Create MultiIndex for rows
index = pd.MultiIndex.from_tuples([('Change in Housing Net Worth,2006-2009', 'coef.'), ('Change in Housing Net Worth,2006-2009', 'std. err'), ('Change in Housing Net Worth,2006-2009', 'p value'),
                                   ('Industry Geographical Herfindahl Index', 'coef.'), ('Industry Geographical Herfindahl Index', 'std. err'), ('Industry Geographical Herfindahl Index', 'p value'),
                                   ('ΔHNW * (Geographical Herfindahl)', 'coef.'), ('ΔHNW * (Geographical Herfindahl)', 'std. err'), ('ΔHNW * (Geographical Herfindahl)', 'p value'),
                                    ('Constant', 'coef.'), ('Constant', 'std. err'),('Constant','p value'),
                                     ('N',''),
                                   ('R-squared','')] ) #names=['Group', 'Number']
# Create multilevel columns
columns = pd.MultiIndex.from_tuples([('Global Trade', 'N','N','N'),
                                    ('Geographical Concentration', 'N','N','N'),
                                    ('Global Trade', 'Y','N','N'),
                                    ('Geographical Concentration', 'Y','N','N'),
                                     ('(5)', 'N','N','N'),
                                     ('(6)', 'N','Y','Y'),
                                     ('(7)', 'N','N','Y')],
                                    names=['Tradable definition used','2-digit 2006 employment share controls?','4-digit Industry Fixed Effects','County Fixed Effects']) #names=['Metric', 'Sub']
# Create the DataFrame
data = [[round(res1.params['netwp_h'],3), round(res2.params['netwp_h'],3), round(res3.params['netwp_h'],3), round(res4.params['netwp_h'],3), round(res5.params['netwp_h'],3), round(res6.params['netwp_h'],3), None],
        [round(res1.bse['netwp_h'],3), round(res2.bse['netwp_h'],3), round(res3.bse['netwp_h'],3), round(res4.bse['netwp_h'],3), round(res5.bse['netwp_h'],3), round(res6.bse['netwp_h'],3), None],
        [round(res1.pvalues['netwp_h'],3), round(res2.pvalues['netwp_h'],3), round(res3.pvalues['netwp_h'],3), round(res4.pvalues['netwp_h'],3), round(res5.pvalues['netwp_h'],3), round(res6.pvalues['netwp_h'],3), None],
        [None, None, None, None, round(res5.params['Iherf'],3), None, None],
        [None, None, None, None, round(res5.bse['Iherf'],3), None, None],
        [None, None, None, None, round(res5.pvalues['Iherf'],3), None, None],
        [None, None, None, None, round(res5.params['netwpherf_b'],3), round(res6.params['netwpherf_b'],3), round(res7.params['netwpherf_b'],3)],
        [None, None, None, None, round(res5.bse['netwpherf_b'],3), round(res6.bse['netwpherf_b'],3), round(res7.bse['netwpherf_b'],3)],
        [None, None, None, None, round(res5.pvalues['netwpherf_b'],3), round(res6.pvalues['netwpherf_b'],3), round(res7.pvalues['netwpherf_b'],3)],
        [round(res1.params['const'],3), round(res2.params['const'],3), round(res3.params['const'],3), round(res4.params['const'],3), round(res5.params['Intercept'],3), None, None],
        [round(res1.bse['const'],3), round(res2.bse['const'],3), round(res3.bse['const'],3), round(res4.bse['const'],3), round(res5.bse['Intercept'],3), None, None],
        [round(res1.pvalues['const'],3), round(res2.pvalues['const'],3), round(res3.pvalues['const'],3), round(res4.pvalues['const'],3), round(res5.pvalues['Intercept'],3), None, None],
        [int(res1.nobs), int(res2.nobs), int(res3.nobs), int(res4.nobs), int(res5.nobs), int(res6.nobs), int(res7.nobs)],
        [round(res1.rsquared,3), round(res2.rsquared,3), round(res3.rsquared,3), round(res4.rsquared,3), round(res5.rsquared,3), round(res6.rsquared,3), round(res7.rsquared,3)]]
table_7 = pd.DataFrame(data, index=index, columns=columns)
table_7.to_excel('Table7.xlsx')

####################################################################################
#Table 8 - Wages and Mobility
####################################################################################
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

data = df_county
# Generate log population change
data["pop0709"] = np.log(data["pop2009"]) - np.log(data["pop2007"])
data.dropna(subset=["Cwage_0709", "netwp_h", "total", "statename", "Cwagehr_Wmean0709","pop0709"],inplace=True)
# Cwage_0709 ~ netwp_h
model1 = smf.wls("Cwage_0709 ~ netwp_h", data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# Cwage_0709 ~ netwp_h + C2D06share*
model2 = smf.wls("Cwage_0709 ~ netwp_h + " + " + ".join([f"C2D06share{i}" for i in range(1, 24)]),
                 data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# Cwagehr_Wmean0709 ~ netwp_h
model3 = smf.wls("Cwagehr_Wmean0709 ~ netwp_h", data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# Cwagehr_Wmean0709 ~ netwp_h + C2D06share*
model4 = smf.wls("Cwagehr_Wmean0709 ~ netwp_h + " + " + ".join([f"C2D06share{i}" for i in range(1, 24)]),
                 data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# pop0709 ~ netwp_h
model5 = smf.wls("pop0709 ~ netwp_h", data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# pop0709 ~ netwp_h + C2D06share*
model6 = smf.wls("pop0709 ~ netwp_h + " + " + ".join([f"C2D06share{i}" for i in range(1, 24)]),
                 data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# Cmovest0709 ~ netwp_h
model7 = smf.wls("Cmovest0709 ~ netwp_h", data=data, weights=data["total"]).fit(cov_type="HC1")

# Cmovest0709 ~ netwp_h + C2D06share*
model8 = smf.wls("Cmovest0709 ~ netwp_h + " + " + ".join([f"C2D06share{i}" for i in range(1, 24)]),
                 data=data, weights=data["total"]).fit(cov_type="HC1")

# Clf_0709 ~ netwp_h
model9 = smf.wls("Clf_0709 ~ netwp_h", data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# Clf_0709 ~ netwp_h + C2D06share*
model10 = smf.wls("Clf_0709 ~ netwp_h + " + " + ".join([f"C2D06share{i}" for i in range(1, 24)]),
                  data=data, weights=data["total"]).fit(
    cov_type="cluster", cov_kwds={"groups": data["statename"]}
)

# Create MultiIndex for rows
index = pd.MultiIndex.from_tuples([('Change in Housing Net Worth,2006-2009', 'coef.'), ('Change in Housing Net Worth,2006-2009', 'std. err'), ('Change in Housing Net Worth,2006-2009', 'p value'),
                                    ('Constant', 'coef.'), ('Constant', 'std. err'),('Constant','p value'),
                                     ('N',''),
                                   ('R-squared','')] ) #names=['Group', 'Number']
# Create multilevel columns
columns = ['Total wage growth, 2007 to 2009, CBP', 'Average Hourly wage growth, 2007 to 2009, ACS','Population growth, 2007-2009', 'Labor force growth, 2007-2009']
# Create the DataFrame
data = [[round(model1.params['netwp_h'],3), round(model3.params['netwp_h'],3), round(model5.params['netwp_h'],3), round(model9.params['netwp_h'],4)],
        [round(model1.bse['netwp_h'],3), round(model3.bse['netwp_h'],3), round(model5.bse['netwp_h'],3), round(model9.bse['netwp_h'],3)],
        [round(model1.pvalues['netwp_h'],3), round(model3.pvalues['netwp_h'],3), round(model5.pvalues['netwp_h'],3), round(model9.pvalues['netwp_h'],3)],
        [round(model1.params['Intercept'],3), round(model3.params['Intercept'],3), round(model5.params['Intercept'],3), round(model9.params['Intercept'],4)],
        [round(model1.bse['Intercept'],3), round(model3.bse['Intercept'],3), round(model5.bse['Intercept'],3), round(model9.bse['Intercept'],3)],
        [round(model1.pvalues['Intercept'],3), round(model3.pvalues['Intercept'],3), round(model5.bse['Intercept'],3), round(model9.bse['Intercept'],3)],
        [int(model1.nobs), int(model3.nobs), int(model5.nobs), int(model9.nobs)],
        [round(model1.rsquared,3), round(model3.rsquared,3), round(model5.rsquared,3), round(model9.rsquared,3)]]
table_8 = pd.DataFrame(data, index=index, columns=columns)
table_8.to_excel('Table8.xlsx')






