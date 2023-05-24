import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Optional


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
                            ('scaler', StandardScaler()), 
                            ('pca', pca_pipe)
                            ]
                        )
        self.all_feats_transform = ColumnTransformer(
            transformers=[('feat_union', feat_union, self.all_cols)]
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]=None):
        """Fits the pipeline to the passed data"""
        self.all_feats_transform.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray):
        """Transforms the passed data"""
        return self.all_feats_transform.transform(X)