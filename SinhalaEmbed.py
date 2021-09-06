import trainer



graph2vec = Graph2Vec(vector_dimensions=128)
graph2vec.parse_graph('data/edge.data', extend_paths=2)
graph2vec.fit(batch_size=1000, max_epochs=1000)
node2vec.model.save_to_file("data/case_embeddings.pkl")