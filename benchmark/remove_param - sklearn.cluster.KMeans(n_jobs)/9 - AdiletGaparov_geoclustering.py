import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class GeoClustering(BaseEstimator, TransformerMixin):
    """Text
        
    Args: 
        name (type): description
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame with new features
    """
    
    def __init__(self, n_clusters=50):
        self.n_clusters = n_clusters
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state = 289))
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X[['longitude','latitude']])             
            
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try:
            labels = self.pipeline.predict(X[['longitude','latitude']]).astype('str')
            X = X.assign(cluster = labels)
            
            return X
            
        except KeyError:
            cols_related = ['longitude','latitude']
            
            cols_error = list(set(cols_related) - set(X.columns))
            raise KeyError('[Clustering] DataFrame does not include the columns:', cols_error)
        
        
        