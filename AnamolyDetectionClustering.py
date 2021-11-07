import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder # label encoding

from sklearn.metrics import mean_squared_error
from scipy.stats import kurtosis


########################## DATA LOAD AND CLEAN ###########################
df = pd.read_csv("Assignment1/data/ML-MATT-CompetitionQT1920_train.csv",encoding='iso-8859-1')
df = df.drop_duplicates()
df["maxUE_UL+DL"] = pd.to_numeric(df["maxUE_UL+DL"], errors='coerce')
labelencoder = LabelEncoder()
df["Time"] = labelencoder.fit_transform(df["Time"])
df["CellName"] = labelencoder.fit_transform(df["CellName"])
df = df.fillna(df.median())

df = resample(df, n_samples=1000, random_state=0)

X = df.drop('Unusual', axis = 1).values
X = StandardScaler().fit_transform(X)
y = df['Unusual'].values

# fig, ax = plt.subplots(figsize=(4,4))
df.hist(bins=30, figsize=(6,6), density=True)
plt.savefig('Assignment3/AD_Charts/AD_Data_Distribution.png')
plt.close()
########################################################################################################################
###################################################### STEP 1 ##########################################################
########################################################################################################################

################### Elbow Chart ############################################
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('Assignment3/AD_Charts/k_means_elbow_chart.png')
plt.close()

#
# ########################## K-MEANS Fit ###########################
kmns = KMeans(n_clusters=5, init='k-means++', algorithm='auto')
kY = kmns.fit_predict(X)
#
# # ############# T-SNE to Visualize clusters #############################
tsne = TSNE(verbose=0, perplexity=20, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X)
#
# # ###################### Visualize Clusters vs Actual ############################################
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('K-Means Clusters')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c = df['Unusual'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual Clusters')
plt.savefig('Assignment3/AD_Charts/k_means_vs_actual.png')
plt.close()
# ################## Expectation Maximization ############################################
bic_arr = []
for i in range(1, 11):
    em = mixture.GaussianMixture(n_components=i, covariance_type='full', random_state = 0)
    em.fit(X)
    bic_arr.append(em.bic(X))

plt.plot(range(1, 11), bic_arr)
plt.title('BIC Chart')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.savefig('Assignment3/AD_Charts/EM_BIC_Chart.png')
plt.close()
# ################## Expectation Maximization ############################################
model = mixture.GaussianMixture(n_components=6, covariance_type='full', random_state=0)
model.fit(X)
labels = model.predict(X)
#
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# GMM
ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=40, cmap='jet')
ax1.set_title('GMM Plot')

# Actual
ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c = df['Unusual'], cmap = "jet", alpha=0.35)
ax2.set_title('Actual clusters')
plt.savefig('Assignment3/AD_Charts/EM_vs_Actual.png')
plt.close()

########################################################################################################################
###################################################### STEP 2 ##########################################################
########################################################################################################################

################################### PCA #############################################
pca = PCA()
pca.fit(X)
################################## Elbow Chart ############################################
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, linewidth=2)
plt.title('SCREE Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Percent of Variance Explained')
plt.savefig('Assignment3/AD_Charts/PCA_Validation.png')
plt.close()
# ################################## PCA Fit #############################################
pca = PCA(n_components=5, random_state=0)
pca.fit(X)
X_pca = pca.transform(X)


# ################################### ICA #############################################
# # Number of components that it takes to reach max kurtosis.
# ################################### Max Kurtosis Chart #################################
arr = []
df.shape[1] -1
for i in range(1,df.shape[1] -1):
    dim_red = FastICA(n_components = i).fit_transform(X)
    kurt = kurtosis(dim_red)
    arr.append(np.mean(kurt))
arr = np.array(arr)
plt.plot(np.arange(1,df.shape[1] - 1),arr)
plt.title('Max Kurtosis Chart')
plt.xlabel('Number of Components')
plt.ylabel('Kurtosis Value')
plt.savefig('Assignment3/AD_Charts/ICA_Validation.png')
plt.close()
# ##################################### Fit ICA #############################################
ica = FastICA(random_state=0, n_components=2) # In this case 26 looks like the best value.
ica.fit(X)
X_ica = ica.transform(X)

# ################################### Random Projection - Reconstruction Error ###########################################
error_array = []
for i in range(1,df.shape[1] -1):
    random_projection = SparseRandomProjection(n_components=i)

    random_projection.fit(X)
    components =  random_projection.components_.toarray() # shape=(5, 11)
    p_inverse = np.linalg.pinv(components.T) # shape=(5, 11)

    #now get the transformed data using the projection components
    reduced_data = random_projection.transform(X) #shape=(4898, 5)
    reconstructed= reduced_data.dot(p_inverse)  #shape=(4898, 11)
    error_array.append(mean_squared_error(X, reconstructed))

# Plot
plt.plot(np.arange(1,df.shape[1] -1),error_array)
plt.title("Reconstruction Error")
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.savefig('Assignment3/AD_Charts/RP_Validation.png')
plt.close()
# ##################################### Fit Random Projection #############################################
random_projection = SparseRandomProjection(n_components=8)
X_randomprojection = random_projection.fit_transform(X)



################################### RFE Chart ###########################################
### RFE is a wrapper feature selection algortihm. This means that for each model selected, the features will be different.
accuracy_array = []
for i in range(1,df.shape[1] -1):
    # create pipeline
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
    model = DecisionTreeClassifier()
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])

    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    accuracy_array.append(np.mean(n_scores))

# # Plot
plt.plot(np.arange(1,df.shape[1] -1),accuracy_array)
# plt.title("Reconstruction Error")
plt.xlabel('Number of Features')
plt.ylabel('Cross Validation Accuracy')
plt.grid()
plt.savefig('Assignment3/AD_Charts/RFE_Validation.png')
plt.close()
# ################################### RFE Fit ###########################################
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
rfe.fit(X,y)
X_RFE = rfe.transform(X)


########################################################################################################################
###################################################### STEP 3.1 ##########################################################
########################################################################################################################
# Kmeans on all 4 dimension reduction algos
###### 1. k-Means - PCA ########
kmeans = KMeans(n_clusters=6, init='k-means++', algorithm='auto')
k_X = kmeans.fit_predict(X_pca)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_pca)

# Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('K-Means w/ PCA')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('K-Means')

ax3.scatter(X_tsne[:,0],X_tsne[:,1],  c = df['Unusual'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax3.set_title('Y Labels')
plt.savefig("Assignment3/AD_Charts/K_means_and_PCA.png")
plt.close()

#
# ####### 2. k-Means - ICA ########
kmeans = KMeans(n_clusters=6, init='k-means++', algorithm='auto', random_state=0)
k_X = kmeans.fit_predict(X_ica)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_ica)

# Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('K-Means w/ ICA')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('K-Means')


plt.savefig("Assignment3/AD_Charts/K_means_and_ICA.png")
plt.close()
#
# ####### 3. k-Means - RandomizedProjection ########
kmeans = KMeans(n_clusters=6, init='k-means++', algorithm='auto', random_state=0)
k_X = kmeans.fit_predict(X_randomprojection)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_randomprojection)
#
# # Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('K-Means w/ RP')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('K-Means')


plt.savefig("Assignment3/AD_Charts/K_means_and_RP.png")
plt.close()
#
# ####### 4. k-Means - RFE ########
kmeans = KMeans(n_clusters=6, init='k-means++', algorithm='auto', random_state=0)
k_X = kmeans.fit_predict(X_RFE)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_RFE)
#
# # Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('K-Means w/ RFE')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('K-Means')

plt.savefig("Assignment3/AD_Charts/K_means_and_RFE.png")
plt.close()
#
#
#
# # ########################################################################################################################
# # ###################################################### STEP 3.2 ##########################################################
# # ########################################################################################################################
# Kmeans on all 4 dimension reduction algos

###### 1. EM - PCA ########
em = mixture.GaussianMixture(n_components=5, covariance_type='full', random_state=0)
k_X = em.fit_predict(X_pca)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_pca)
#
# # Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('EM w/ PCA')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('EM')

plt.savefig("Assignment3/AD_Charts/EM_and_PCA.png")
plt.close()
# ####### 2. EM - ICA ########
em = mixture.GaussianMixture(n_components=5, covariance_type='full', random_state=0)
k_X = em.fit_predict(X_ica)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_ica)

# Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('EM w/ ICA')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('EM')

plt.savefig("Assignment3/AD_Charts/EM_and_ICA.png")
plt.close()
# ####### 3. EM - RandomizedProjection ########
em = mixture.GaussianMixture(n_components=5, covariance_type='full', random_state=0)
k_X = em.fit_predict(X_randomprojection)

# T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_randomprojection)

# Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('EM w/ RP')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('EM')


plt.savefig("Assignment3/AD_Charts/EM_and_RP.png")
plt.close()
# ####### 4. EM - RFE ########
em = mixture.GaussianMixture(n_components=5, covariance_type='full', random_state=0)
k_X = em.fit_predict(X_RFE)
#
# # T-SNE to Visualize clusters
tsne = TSNE(verbose=0, perplexity=40, n_iter= 4000, random_state=0)
X_tsne = tsne.fit_transform(X_RFE)

# Visualize Clusters vs Actual
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
ax1.scatter(X_tsne[:,0],X_tsne[:,1],  c=k_X, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('EM w/ RFE')

ax2.scatter(X_tsne[:,0],X_tsne[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('EM')

plt.savefig("Assignment3/AD_Charts/EM_and_RFE.png")
plt.close()


