#         self._add_module_meta()
#         self._add_month()

    
    def get_cat_feats(self) -> List[str]:
        return self.cat_feats
    
    
        def _add_module_meta(self) -> None:
        """Add metadata of generator module."""
        for feat, meta_map in MODULE_META.items():
            self._df[feat] = self._df["Module"].map(meta_map)
    
    def _add_month(self) -> None:
        self._df["Month"] = pd.to_datetime(self._df["Date"], format="%Y-%m-%d").dt.month
        from sklearn.preprocessing import LabelEncoder
        enc = LabelEncoder()
        self._df["Module"] = enc.fit_transform(self._df["Module"])
        self.cat_feats = ["Month", "Module"]