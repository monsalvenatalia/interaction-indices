# Game-theoretic Interaction Indices for link prediction and community detection

## Introduction

Social networks are defined as structures of interactions in which nodes represent actors and
edges model the relationships between them. The rise of online social networks in recent decades
has provided us with large amounts of data, enabling more in-depth research into their structure
and dynamics.

From the perspective of data mining, two primary types of data can be distinguished. One of
them is linkage-based and structural analysis, which allows us to identify important communities
and connections between nodes. This analysis can be approached through the concept of **interaction**,
as defined in **cooperative game theory**. Interaction indices, derived from solution concepts in game
theory, will be employed as measures of similarity between nodes, providing an effective tool for
**link prediction** and **community detection**.

Based on these indices, we will implement algorithms (inspired by existing pseudocode from related literature) for their computation, 
and apply them to the problems of link prediction and community detection in real-world datasets. Finally, the obtained results will be compared with other local
similarity measures to assess the efficiency of our approach.

## Link Prediction
In numerous scenarios, the formation of links between nodes can be inferred from the underlying structural properties of the network.
This project adopts a purely topological approach by modeling the network as a cooperative game, where the characteristic function is defined in an *endogenous manner*.

Link prediction techniques generally assign a similarity score to each pair of unconnected nodes, indicating the **likelihood** that a link will form between them in the future. 
These approaches are grounded in the assumption that structurally proximate nodes are more inclined to establish connections. The result is a ranked list of node pairs, where those with the highest scores are deemed the most probable candidates for future links.

## Community Detection
Real-world networks often exhibit a high degree of organization and heterogeneity. A network is said to have community structure when it contains groups
of nodes that are **densely connected internally** and **sparsely connected to the rest** of the network.

Community detection aims to extract topological information in order to find a partition of the nodes—in our case, a *disjoint* one. In this project,
we apply a hierarchical agglomerative clustering algorithm, using *complete linkage* as the merging criterion. Unlike standard implementations that rely on distance, we define similarity
between communities using a game-theoretic interaction index.





