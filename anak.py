from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo': 'a', 'bar': 'b'}, {'foo': 'b', 'baz': 'c'}]
# D = [{'foo': ['a']}, {'foo': ['b'], 'baz': ['c']}]

X = v.fit_transform(D)
print(X)
print(X.shape) # (2,4) => 2 set with 4 dimension (4 uniq value)