# Koha Neural Network

This is an ongoing research repository for the development of the Koha neural network.

The Koha neural network is a new type of artificial neural network inspired by the Koha theory of biological memory.

The idea of the Koha neural network is to enable cheap horizontal scaling for large neural networks. Listed are some interesting (and yet to be proven) attributes of the Koha neural network:
- Conventional neural networks are structured as directed graphs. The Koha neural network is structured as an undirected graph.
- Gradients within the Koha neural network do not flow between Koha blocks. That is, each Koha block is trained independently.
- There is no sequential dependency for information propagation between Koha blocks. This enables Koha blocks to be trained in parallel.
- Furthermore, minimal data is shared between Koha blocks. Because of the parallel, as well as compartmentalized nature of Koha blocks, Koha neural networks can be trained effectively within cluster of weaker machines. The architecture is designed for horizontal scaling.

# Useful resources
- [The Koha model paper discussion](https://youtu.be/nR0SBUOoLRM)
- [The Koha model draft paper](https://arxiv.org/pdf/2109.02124.pdf)
- [The Koha neural network draft idea discussion](https://youtu.be/xsFKAnmV-IY)