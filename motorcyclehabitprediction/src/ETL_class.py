import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, BaseEstimator


class DataTransformer(BaseEstimator, TransformerMixin):       
    """Class that does the data transformation"""
    def __init__(self, all_cols, pca_out_num=10):
        self.pca_out_num = pca_out_num
        self.all_cols = all_cols

        pca_pipe = Pipeline(steps=[
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=self.pca_out_num))
            ]
        )

        feat_union = FeatureUnion(transformer_list=[
                            ('scaler', StandardScaler(),), 
                            ('pca', pca_pipe)
                            ]
                        )
        self.all_feats_transform = ColumnTransformer(
            transformers=[('feat_union', feat_union, self.all_cols)]
        )

    def fit(self, X, y=None):
        self.all_feats_transform.fit(X, y)
        return self
    
    def transform(self, X):
        return self.all_feats_transform.transform(X)